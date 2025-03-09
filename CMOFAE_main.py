import sys
sys.path.append("./")
sys.path.append("./sam2_repo")

from torchvision.models import resnet50, ResNet50_Weights
import geatpy as ea
from CMOFAE_Problem import CMOFAEProblem
from moea_NSGA2_templet import moea_NSGA2_templet
from concept_segmentation import concept_segmentation_pipeline
from concept_segmentation import load_sam2
import torch
from PIL import Image
from result_process import *
import numpy as np

def run(parameters):
    problem = CMOFAEProblem(M=2, Dim=parameters['Dim'], lb=parameters['lb_value'], ub=parameters['ub_value'],
                                  model=parameters['model'], img=parameters['img'], masks=parameters['masks'],
                                  pred_class=parameters['pred_class'],
                                  explained_img_prob_score=parameters['explained_img_prob_score'],
                                  black_img_prob_score=parameters['black_img_prob_score'],
                                  delta_scores=parameters['delta_scores'],)

    algorithm = moea_NSGA2_templet(problem,
                                      ea.Population(Encoding='RI', NIND=parameters['pop_size']),
                                      MAXGEN=parameters['max_gen'],
                                      logTras=0)
    # Do the optimization
    res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=True, saveFlag=False)

    return res['ObjV'], res['Vars'], res['executeTime']

def main(device):
    explained_img_path = 'ILSVRC2012_test_00000005_TV.JPEG'

    # ============================== Step 1: load the explained model f (ResNet-50) ==============================
    weights = ResNet50_Weights.DEFAULT
    explained_model = resnet50(weights=weights)
    if device.type == "cuda":
        explained_model.to("cuda")
    explained_model.eval()

    # ==================================== Step 2: load the explained image x ====================================
    img = Image.open(explained_img_path)
    preprocess = weights.transforms()
    explained_img = preprocess(img).unsqueeze(0)
    if device.type == "cuda":
        explained_img = explained_img.to("cuda")

    # =========== Step 3: segmentation of the explained image using SAM2 and post-processing techniques ==========
    min_segment = 64
    contain_threshold = 0.90  # the threshold of concept inclusion \theta
    mask_generator = load_sam2(min_segment, device=device)
    masks = concept_segmentation_pipeline(explained_img_path, mask_generator=mask_generator, contain_threshold=contain_threshold)
    show_segmentation_result(img, masks)

    # ======================================== Step 4: paramerter setting ========================================
    class_id = explained_model(explained_img).squeeze(0).softmax(0).argmax().item()  # Predicted label
    explained_img_prob_score = explained_model(explained_img).squeeze(0).softmax(0)[class_id].item()  # Predicted probability

    # The prediction for the all-black image, i.e., w_0
    black_img = torch.zeros_like(explained_img)
    if device.type == "cuda":
        black_img = black_img.to("cuda")
    black_img_prob_score = explained_model(black_img).squeeze(0).softmax(0)[class_id].item()  # Predicted probability of pure black images

    # lower bound and upper bound of the decision variables
    lb_value = -1.5 * np.abs(explained_img_prob_score)
    ub_value = 1.5 * np.abs(explained_img_prob_score)

    # Calculate the predicted probabilities after perturbing the image, which will be used when evaluating the faithfulness metric \mu_c
    perturbations_img = np.tile(np.array(img).copy(), (len(masks), 1, 1, 1))
    for i in range(len(masks)):
        temp_mask = torch.from_numpy(masks[i]['segmentation']).unsqueeze(2).expand(-1, -1, 3)
        perturbations_img[i, temp_mask] = 0
    preprocess_perturbations_img = []
    for i in range(len(perturbations_img)):
        preprocess_perturbations_img.append(preprocess(Image.fromarray(perturbations_img[i, :, :, :])))
    preprocess_perturbations_img = torch.tensor(np.array(preprocess_perturbations_img))
    if device.type == 'cuda':
        preprocess_perturbations_img = preprocess_perturbations_img.to('cuda')
    perturbations_scores = explained_model(preprocess_perturbations_img).squeeze(0).softmax(0)[:, class_id]
    delta_scores = (explained_img_prob_score - perturbations_scores).cpu().detach().to(torch.float32).numpy()

    # Setting population size, maximum number of iterations, crossover and mutation probabilities
    max_gen = 1000
    pop_size = 100
    mutation_p, crossover_p = 1, 1

    parameters = {'model': explained_model, 'img': explained_img, 'masks': masks, 'pred_class': class_id,
                  'explained_img_prob_score': explained_img_prob_score,
                  'black_img_prob_score': black_img_prob_score, 'lb_value': lb_value, 'ub_value': ub_value,
                  'delta_scores': delta_scores, 'max_gen': max_gen, 'pop_size': pop_size, 'mutation_p': mutation_p,
                  'crossover_p': crossover_p, 'Dim': len(masks)}

    # ========================================= Step 5: Running CMOFAE =========================================
    F, X, run_time = run(parameters)

    show_all_solution(F)
    show_explanation_result(F, X, img, masks)

    save_result(F, X)

if __name__ == '__main__':
    # Specified CPU/GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rand_seed = 0
    np.random.seed(rand_seed)
    main(device)