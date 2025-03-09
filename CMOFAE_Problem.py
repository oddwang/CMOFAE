# -*- coding: utf-8 -*-
import numpy as np
import torch

import geatpy as ea

class CMOFAEProblem(ea.Problem):
    def __init__(self, M=2, Dim=None, lb=None, ub=None, model=None, img=None, masks=None, pred_class=None,
                 explained_img_prob_score=None, black_img_prob_score=None, delta_scores=None):
        name = 'CMOFAE_Problem'  # Problem's name.
        maxormins = [1] * M  # All objects are need to be minimized.
        varTypes = [0] * Dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [lb] * Dim  # The lower bound of each decision variable.
        ub = [ub] * Dim  # The upper bound of each decision variable.
        lbin = [1] * Dim  # Whether the lower boundary is included.
        ubin = [1] * Dim  # Whether the upper boundary is included.
        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        self.model = model
        self.img = img
        self.masks = masks
        self.pred_class = pred_class
        self.explained_img_prob_score = explained_img_prob_score
        self.black_img_prob_score = black_img_prob_score
        self.best_faithful = None
        self.best_faithful_rank = None
        self.delta_scores = delta_scores

    def cal_faithfulness(self, exp):
        return -torch.corrcoef(torch.stack([exp, torch.tensor(self.delta_scores)]))[0, 1]

    def aimFunc(self, pop):
        x = pop.Phen
        # calculate faithfulness \mu_F
        faithfulness = []
        for row in x:
            faithfulness.append(np.corrcoef(row, self.delta_scores)[0, 1])
        faithfulness = np.array(faithfulness)

        # calculate complexity \mu_C
        fractions = np.abs(x) / np.sum(np.abs(x), axis=1)[:, np.newaxis]
        complexity = -np.nansum(fractions * np.log(fractions), axis=1)

        pop.ObjV = np.column_stack([-faithfulness, complexity])

        cur_faithful_idx = np.argmax(faithfulness)
        if self.best_faithful == None or faithfulness[cur_faithful_idx] > self.best_faithful:
            self.best_faithful = faithfulness[cur_faithful_idx]
            self.best_faithful_rank = np.argsort(np.argsort(x[cur_faithful_idx, :]))

    def partial_training(self, pop):
        # find the most faithfulness individual and then execute partial training
        idx = np.argmin(pop.ObjV[:, 0])

        params_to_update = torch.tensor(pop.Chrom[idx, :])
        params_to_update.requires_grad = True
        loss = self.cal_faithfulness(params_to_update)

        optimizer = torch.optim.AdamW(
            [params_to_update],
            lr=0.01,
            weight_decay=0.05,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pop.Chrom[idx, :] = params_to_update.detach().numpy()