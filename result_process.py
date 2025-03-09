import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import datetime
import os
import pandas as pd

def show_segmentation_masks(anns, borders=True, show_number=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    ann_index_map = {id(ann): idx for idx, ann in enumerate(anns)}
    for idx, ann in enumerate(sorted_anns, start=1):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
            M = cv2.moments(m.astype(np.uint8))
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = 0, 0

        original_idx = ann_index_map[id(ann)] + 1

        if show_number:
            ax.text(cX, cY, str(original_idx), color='white', fontsize=16, ha='center', va='center', fontweight='bold')

    ax.imshow(img)

def show_explanation_masks(anns, mask_scores, borders=True, show_number=True, alpha=0.8, vmin=-2, vmax=2):
    sorted_anns = anns

    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    midpoint = 0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=midpoint)
    cmap = cm.get_cmap('bwr')

    ann_index_map = {id(ann): idx for idx, ann in enumerate(anns)}

    for idx, ann in enumerate(sorted_anns, start=1):
        m = ann['segmentation']
        score = mask_scores[ann_index_map[id(ann)]]

        color = cmap(norm(score))
        color = list(color[:3]) + [alpha]
        img[m] = color

        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 0, 0.5), thickness=1)

        M = cv2.moments(m.astype(np.uint8))
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = 0, 0
        original_idx = ann_index_map[id(ann)] + 1

        if show_number:
            ax.text(cX, cY, str(original_idx), color='white', fontsize=16, ha='center', va='center', fontweight='bold')

    ax.imshow(img)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    tick_values = [norm.vmin, norm.vmin / 2, 0.0, norm.vmax / 2, norm.vmax]
    cbar.set_ticks(tick_values)
    cbar.set_label('Mask Scores', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

def show_segmentation_result(image, masks):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_segmentation_masks(masks)
    plt.axis('off')
    plt.title('Segmentation Result', fontsize=48)
    plt.show()

def show_all_solution(F):
    plt.scatter(-F[:, 0], F[:, 1], c='red', s=100)

    plt.xlabel('Faithfulness $\mu_F$', fontsize=16)
    plt.ylabel('Complexity $\mu_C$', fontsize=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('Evaluation Results for All Obtained Solutions', fontsize=20)
    plt.show()

def show_explanation_result(F, X, img, masks):
    most_faithful_idx = np.argmin(F[:, 0])
    most_incomplexity_idx = np.argmin(F[:, 1])
    mid_value = (np.max(F[:, 1]) + np.min(F[:, 1])) / 2
    mid_idx = np.argmin(np.abs(F[:, 1] - mid_value))

    vmax = np.max([X[most_faithful_idx], X[most_incomplexity_idx], X[mid_idx]])
    vmin = np.min([X[most_faithful_idx], X[most_incomplexity_idx], X[mid_idx]])

    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    show_explanation_masks(masks, X[most_faithful_idx, :], show_number=False, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title('Most Faithfulness Explanation Result', fontsize=48)
    plt.show()

    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    show_explanation_masks(masks, X[mid_idx, :], show_number=False, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title('Moderately Well-performing Explanation Results', fontsize=48)
    plt.show()

    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    show_explanation_masks(masks, X[most_incomplexity_idx, :], show_number=False, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title('Least Complexity Explanation Result', fontsize=48)
    plt.show()

def save_result(F, X):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    folder_name = f"./Result/{current_time}"
    os.makedirs(folder_name, exist_ok=True)
    F_path = os.path.join(folder_name, "F.csv")
    X_path = os.path.join(folder_name, "X.csv")
    pd.DataFrame(F).to_csv(F_path, index=False)  # 保存 F 矩阵
    pd.DataFrame(X).to_csv(X_path, index=False)  # 保存 X 矩阵