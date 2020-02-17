import torch.nn as nn
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN
# self.word2idx['<UNK>'] = 1


class DecoderRNN(BaseRNN):

    def __init__(self, vocab_size, embedding, embed_size, pemsize, sos_id, eos_id,
                 unk_id, max_len=100, n_layers=1, rnn_cell='gru',
                 bidirectional=True, input_dropout_p=0, dropout_p=0,
                 lmbda=1.5, USE_CUDA = torch.cuda.is_available(), beam_size=5):
        hidden_size = embed_size

        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.embedding = embedding
        self.lmbda = lmbda
        self.beam_size = beam_size
        self.device = torch.device("cuda:0" if USE_CUDA and torch.cuda.is_available() else "cpu")
        #directions
        self.Wh = nn.Linear(hidden_size * 2, hidden_size)
        #output
        self.V = nn.Linear(hidden_size * 3, self.output_size)
        #params for attention
        self.Wih = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder input
        self.Wfh = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder field
        self.Ws = nn.Linear(hidden_size, hidden_size)  # for obtaining e from current state
        self.w_c = nn.Linear(1, hidden_size)  # for obtaining e from context vector
        self.v = nn.Linear(hidden_size, 1)
        # parameters for p_gen
        self.w_ih = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_fh = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_s = nn.Linear(hidden_size, 1)    # for changing hidden state into a scalar
        self.w_x = nn.Linear(embed_size, 1)     # for changing input embedding into a scalar
        # parameters for self attention
        self_size = pemsize * 2  # hidden_size +
        self.wp = nn.Linear(self_size, self_size)
        self.wc = nn.Linear(self_size, self_size)
        self.wa = nn.Linear(self_size, self_size)

    def get_matrix(self, encoderp):
        tp = torch.tanh(self.wp(encoderp))
        tc = torch.tanh(self.wc(encoderp))
        f = tp.bmm(self.wa(tc).transpose(1, 2))
        return F.softmax(f, dim=2)

    def self_attn(self, f_matrix, encoderi, encoderf):
        c_contexti = torch.bmm(f_matrix, encoderi)
        c_contextf = torch.bmm(f_matrix, encoderf)
        return c_contexti, c_contextf

    def decode_step(self, input_ids, coverage, _h, enc_proj, batch_size, max_enc_len,
                    enc_mask, c_contexti, c_contextf, embed_input, max_source_oov):
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        cov_proj = self.w_c(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
        e_t = self.v(torch.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))

        # mask to -INF before applying softmax
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
        attn_scores.data.masked_fill_(enc_mask.data.bool(), -10000)
        attn_scores = F.softmax(attn_scores, dim=1)

        contexti = attn_scores.unsqueeze(1).bmm(c_contexti).squeeze(1)
        contextf = attn_scores.unsqueeze(1).bmm(c_contextf).squeeze(1)

        # output proj calculation
        p_vocab = F.softmax(self.V(torch.cat((_h, contexti, contextf), 1)), dim=1)
        # p_gen calculation
        p_gen = torch.sigmoid(self.w_ih(contexti) + self.w_fh(contextf) + self.w_s(_h) + self.w_x(embed_input))
        p_gen = p_gen.view(-1, 1)
        weighted_Pvocab = p_vocab * p_gen
        weighted_attn = (1-p_gen) * attn_scores

        if max_source_oov > 0:
            # create OOV (but in-article) zero vectors
            ext_vocab = torch.zeros(batch_size, max_source_oov)
            ext_vocab=ext_vocab.to(self.device)
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        del weighted_Pvocab       # 'Recheck OOV indexes!'
        # scatter article word probs to combined vocab prob.
        combined_vocab = combined_vocab.scatter_add(1, input_ids, weighted_attn)
        return combined_vocab, attn_scores

    def forward(self, max_source_oov=0, targets=None, targets_id=None, input_ids=None,
                enc_mask=None, encoder_hidden=None, encoderi=None, encoderf=None,
                encoderp=None, teacher_forcing_ratio=None, w2fs=None, fig=False, beam=False):

        targets, batch_size, max_length, max_enc_len = self._validate_args(targets, encoder_hidden, encoderi, teacher_forcing_ratio)

        decoder_hidden = self._init_state(encoder_hidden)
        if beam:            
            return self.beam_search(targets, max_length, decoder_hidden, max_enc_len, max_source_oov, input_ids, enc_mask, encoderi, encoderf,
                encoderp, w2fs)
        coverage = torch.zeros(batch_size, max_enc_len).to(self.device)
        enci_proj = self.Wih(encoderi.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        encf_proj = self.Wfh(encoderf.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        f_matrix = self.get_matrix(encoderp)
        enc_proj = enci_proj + encf_proj

        # get position attention scores
        c_contexti, c_contextf = self.self_attn(f_matrix, encoderi, encoderf)
        if teacher_forcing_ratio:
            embedded = self.embedding(targets)
            embed_inputs = self.input_dropout(embedded)
            # coverage initially zero
            dec_lens = (targets > 0).float().sum(1)
            lm_loss, cov_loss = [], []
            hidden, _ = self.rnn(embed_inputs, decoder_hidden)
            # step through decoder hidden states
            for _step in range(max_length):
                _h = hidden[:, _step, :]
                target_id = targets_id[:, _step+1].unsqueeze(1)
                embed_input = embed_inputs[:, _step, :]

                combined_vocab, attn_scores = self.decode_step(input_ids, coverage, _h, enc_proj, batch_size,
                                                               max_enc_len, enc_mask, c_contexti, c_contextf,
                                                               embed_input, max_source_oov)
                # mask the output to account for PAD
                target_mask_0 = target_id.ne(0).detach()
                output = combined_vocab.gather(1, target_id).add_(sys.float_info.epsilon)
                lm_loss.append(output.log().mul(-1) * target_mask_0.float())

                coverage = coverage + attn_scores

                # Coverage Loss
                # take minimum across both attn_scores and coverage
                _cov_loss, _ = torch.stack((coverage, attn_scores), 2).min(2)
                cov_loss.append(_cov_loss.sum(1))
            # add individual losses
            total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens) + self.lmbda * \
                torch.stack(cov_loss, 1).sum(1).div(dec_lens)
            return total_masked_loss
        else:
            return self.evaluate(targets, batch_size, max_length, max_source_oov, c_contexti, c_contextf, f_matrix,
                                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig)

    def evaluate(self, targets, batch_size, max_length, max_source_oov, c_contexti, c_contextf, f_matrix,
                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig):
        lengths = np.array([max_length] * batch_size)
        decoded_outputs = []
        if fig:
            attn = []
        embed_input = self.embedding(targets)
        # step through decoder hidden states
        for _step in range(max_length):
            _h, decoder_hidden = self.rnn(embed_input, decoder_hidden)
            combined_vocab, attn_scores = self.decode_step(input_ids, coverage,
                                                           _h.squeeze(1), enc_proj, batch_size, max_enc_len, enc_mask,
                                                           c_contexti, c_contextf, embed_input.squeeze(1),
                                                           max_source_oov)
            # not allow decoder to output UNK
            combined_vocab[:, self.unk_id] = 0
            symbols = combined_vocab.topk(1)[1]

            if fig:
                attn.append(attn_scores)
            decoded_outputs.append(symbols.clone())
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > _step) & eos_batches) != 0
                lengths[update_idx] = len(decoded_outputs)
            # change unk to corresponding field
            for i in range(symbols.size(0)):
                w2f = w2fs[i]
                if symbols[i].item() > self.vocab_size-1:
                    symbols[i] = w2f[symbols[i].item()]
            # symbols.masked_fill_((symbols > self.vocab_size-1), self.unk_id)
            embed_input = self.embedding(symbols)
            coverage = coverage + attn_scores
        if fig:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), f_matrix[0], \
                   torch.stack(attn, 1).squeeze(2)[0]
        else:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist()

    def beam_search(self, targets, max_length, decoder_hidden, max_enc_len, max_source_oov, input_ids, enc_mask, encoderi, encoderf,
                encoderp, w2fs):
        # the code is very similar to the forward function
        w2f = w2fs[0]

        ### YOUR CODE HERE (~1 Lines) 
        ### Initialize coverage vector 
        batch_size = 1
        coverage = torch.zeros(batch_size, max_enc_len).to(self.device)
        ### END YOUR CODE    
          
        # results --> list of all ouputs terminated with stop tokens and of minimal length
        results = []
        # all_hyps --> list of current beam hypothesis. start with base initial hypothesis
        all_hyps = [Hypothesis([self.sos_id], decoder_hidden, coverage, 0)]
        # start decoding

        ### YOUR CODE HERE (~4 Lines) 
        ### Initialize enci_proj, encf_proj, f_matrix, enc_proj vector 
        enci_proj = self.Wih(encoderi.view(batch_size * max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        encf_proj = self.Wfh(encoderf.view(batch_size * max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        f_matrix = self.get_matrix(encoderp)
        enc_proj = enci_proj + encf_proj
        ### END YOUR CODE      

        ### YOUR CODE HERE (~1 Lines) 
        # get position attention scores
        c_contexti, c_contextf = self.self_attn(f_matrix, encoderi, encoderf)
        ### END YOUR CODE      

        embed_input = self.embedding(targets)
        # step through decoder hidden states
        for _step in range(max_length):
            # after first step, input is of batch_size=curr_beam_size
            # curr_beam_size <= self.beam_size due to pruning of beams that have terminated
            # adjust enc_states and init_state accordingly

            _h, decoder_hidden = self.rnn(embed_input, decoder_hidden)

            ### YOUR CODE HERE (~1 Lines) 
            ### get current beam size curr_beam_size
            curr_beam_size = embed_input.size(0)
            ### END YOUR CODE      
            
            combined_vocab, attn_scores = self.decode_step(input_ids, coverage,
                                                           _h.squeeze(1), enc_proj, curr_beam_size, max_enc_len, enc_mask,
                                                           c_contexti, c_contextf, embed_input.squeeze(1),
                                                           max_source_oov)
            combined_vocab[:, self.unk_id] = 0
            all_hyps, symbols, results, decoder_hidden, coverage = self.getOverallTopk(combined_vocab,
                                                                                attn_scores, coverage, all_hyps, results,
                                                                                decoder_hidden.squeeze(0))
            # change unk to corresponding field
            for i in range(symbols.size(0)):
                if symbols[i].item() > self.vocab_size-1:
                    symbols[i] = w2f[symbols[i].item()]
            embed_input = self.embedding(symbols)
            curr_beam_size = embed_input.size(0)
            if embed_input.size(0) > encoderi.size(0):
                c_contexti = c_contexti.expand(curr_beam_size, max_enc_len, -1).contiguous()
                c_contextf = c_contextf.expand(curr_beam_size, max_enc_len, -1).contiguous()
                enc_proj = enc_proj.expand(curr_beam_size, max_enc_len, -1).contiguous()
                enc_mask = enc_mask.expand(curr_beam_size, -1).contiguous()
                input_ids = input_ids.expand(curr_beam_size, -1).contiguous()
        
        if len(results) > 0:
            candidates = results
        else:
            candidates = all_hyps
        all_outputs = sorted(candidates, key=lambda x:x.survivability, reverse=True)
        return all_outputs[0].full_prediction
            


    def getOverallTopk(self, vocab_probs, attn_scores, coverage, all_hyps, results, decoder_hidden):
        new_decoder_hidden, new_coverage = [], []
        new_vocab_probs = []
        for i, hypo in enumerate(all_hyps):
            curr_vocab_probs = vocab_probs[i]
            new_vocab_probs.append(curr_vocab_probs.unsqueeze(0))
        vocab_probs = torch.cat(new_vocab_probs, 0)
        coverage += attn_scores
        # return top-k values i.e. top-k over all beams i.e. next step input ids
        # return hidden, cell states corresponding to topk
        probs, inds = vocab_probs.topk(k=self.beam_size, dim=1)
        probs = probs.log()
        candidates = []
        assert len(all_hyps) == probs.size(0), '# Hypothesis and log-prob size dont match'
        ### YOUR CODE HERE (~4 Lines) 
        ### cycle through all hypothesis in full beam
        for i, hypo in enumerate(probs.tolist()):
            for j, _ in enumerate(hypo):
                new_cand = all_hyps[i].extend(token_id=inds[i, j].item(),
                                              hidden_state=decoder_hidden[i].unsqueeze(0),
                                              coverage=coverage[i].unsqueeze(0),
                                              log_prob=probs[i,j])
                candidates.append(new_cand)
        ### END YOUR CODE      
        # sort in descending order
        candidates = sorted(candidates, key=lambda x:x.survivability, reverse=True)
        # print('len of candidiates: ', len(candidates))
        new_beam, next_inp = [], []
        # prune hypotheses and generate new beam
        for h in candidates:
            if h.full_prediction[-1] == self.eos_id and len(h.full_prediction) > 2:
                # weed out small sentences that likely have no meaning
                if len(h.full_prediction) >= 15:
                    results.append(h)
            else:
                new_beam.append(h)
                next_inp.append(h.full_prediction[-1])
                new_decoder_hidden.append(h.hidden_state)
                new_coverage.append(h.coverage)
            if len(new_beam) == self.beam_size:
                break
        # print('len of beam: ', len(new_beam))
        assert len(new_beam) >= 1, 'Non-existent beam'
        # print(next_inp)
        return new_beam, torch.LongTensor(next_inp).to(self.device).view(-1, 1), results, \
               torch.cat(new_decoder_hidden, 0).unsqueeze(0), torch.cat(new_coverage, 0)

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            h = self.Wh(h)
        return h

    def _validate_args(self, targets, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        else:
            max_enc_len = encoder_outputs.size(1)
        # inference batch size
        if targets is None and encoder_hidden is None:
            batch_size = 1
        else:
            if targets is not None:
                batch_size = targets.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default targets and max decoding length
        if targets is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")
            # torch.set_grad_enabled(False)
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            targets = targets.to(self.device)
            max_length = self.max_length
        else:
            max_length = targets.size(1) - 1     # minus the start of sequence symbol

        return targets, batch_size, max_length, max_enc_len
    

class Hypothesis(object):
    def __init__(self, token_id, hidden_state, coverage, log_prob):
        self._h = hidden_state
        self.log_prob = log_prob
        self.hidden_state = hidden_state
        self.coverage = coverage.detach()
        self.full_prediction = token_id # list
        self.survivability = self.log_prob/ float(len(self.full_prediction))

    def extend(self, token_id, hidden_state, coverage, log_prob):
        return Hypothesis(token_id= self.full_prediction + [token_id],
                          hidden_state=hidden_state,
                          coverage=coverage.detach(),
                          log_prob= self.log_prob + log_prob)