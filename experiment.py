# mammary gland- 39 HE는 실험에서 제외
# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import numpy as np

import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
matcher = LightGlue(features="superpoint").eval().to(device)
# "38-ER", "40-PR", '10','25',
stack = ["38-ER", "40-PR","36-CNEU"]

stack2 = ['50']
stack3 = [1024]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kabsch_algorithm(P, Q):
    """
    Kabsch algorithm for finding the optimal rotation matrix.
    Args:
        P: (N, 2) tensor of points in the first set (source points)
        Q: (N, 2) tensor of points in the second set (target points)
        
    Returns:
        R: (2, 2) tensor of rotation matrix
        t: (2,) tensor of translation vector
    """
    P_centered = P - P.mean(dim=0)
    Q_centered = Q - Q.mean(dim=0)

    H = torch.matmul(P_centered.T, Q_centered)

    U, _, V = torch.svd(H)

    R = torch.matmul(V, U.T)
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = torch.matmul(V, U.T)
    t = Q.mean(dim=0) - torch.matmul(P.mean(dim=0), R)

    return R, t
# for j in stack3:
#     extractor = SuperPoint(max_num_keypoints=j).eval().to(device)  # load the extractor
#     for k in stack2:
#         for i in stack:
#             image_0 = "mammary-gland_1/scale-"+k+"pc/s1_37-HE_A4926-4L.jpg"
#             image_1 = "mammary-gland_1/scale-"+k+"pc/s1_"+i+"_A4926-4L.jpg"
#             image0 = load_image(image_0)
#             image1 = load_image(image_1)

#             feats0 = extractor.extract(image0.to(device))
#             feats1 = extractor.extract(image1.to(device))
#             matches01 = matcher({"image0": feats0, "image1": feats1})
#             feats0, feats1, matches01 = [
#                 rbd(x) for x in [feats0, feats1, matches01]
#             ]  # remove batch dimension

#             kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
#             m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

#             axes = viz2d.plot_images([image0, image1])
#             viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", ps=0.25,lw=0.2)
#             # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=7)
#             viz2d.save_plot("./output/lightglue/superpoint/mammary-gland_1/scale-"+k+"pc/matching/keypoints_"+f'{j}'+"_37-HE_"+i+".jpg", dpi=300)
#             plt.close()
#             kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
#             viz2d.plot_images([image0, image1])
#             viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=0.25)
#             viz2d.save_plot("./output/lightglue/superpoint/mammary-gland_1/scale-"+k+"pc/visualization/keypoints_"+f'{j}'+"_37-HE_"+i+".jpg", dpi=300)
#             plt.close()
# stack = [ "62-ER"]
# for j in stack3:
#     extractor = SuperPoint(max_num_keypoints=j).eval().to(device)  # load the extractor
#     for k in stack2:
#         for i in stack:
#             image_0 = "mammary-gland_2/scale-"+k+"pc/s2_61-HE_A4926-4L.jpg"
#             image_1 = "mammary-gland_2/scale-"+k+"pc/s2_"+i+"_A4926-4L.jpg"
#             image0 = load_image(image_0)
#             image1 = load_image(image_1)

#             feats0 = extractor.extract(image0.to(device))
#             feats1 = extractor.extract(image1.to(device))
#             matches01 = matcher({"image0": feats0, "image1": feats1})
#             feats0, feats1, matches01 = [
#                 rbd(x) for x in [feats0, feats1, matches01]
#             ]  # remove batch dimension

#             kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
#             m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

#             axes = viz2d.plot_images([image0, image1])
#             viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", ps=0.25,lw=0.2)
#             # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=7)
#             viz2d.save_plot("./output/lightglue/superpoint/mammary-gland_2/scale-"+k+"pc/matching/keypoints_"+f'{j}'+"_37-HE_"+i+".jpg", dpi=300)
#             plt.close()
#             kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
#             viz2d.plot_images([image0, image1])
#             viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=0.25)
#             viz2d.save_plot("./output/lightglue/superpoint/mammary-gland_2/scale-"+k+"pc/visualizaiton/keypoints_"+f'{j}'+"_37-HE_"+i+".jpg", dpi=300)
#             plt.close()
stack = [ "proSPC-4", "Ki67-7", "Cc10-5"]            
# for l in range(1,4):
#     for j in stack3:
#         extractor = DISK(max_num_keypoints=j).eval().to(device)  # load the extractor
#         for k in stack2:
#             for i in stack:
#                 image_0 = f"lung-lesion_{l}/scale-"+k+f"pc/29-041-Izd2-w35-He-les{l}.jpg"
#                 image_1 = f"lung-lesion_{l}/scale-"+k+"pc/29-041-Izd2-w35-"+i+f"-les{l}.jpg"
#                 image0 = load_image(image_0)
#                 image1 = load_image(image_1)

#                 feats0 = extractor.extract(image0.to(device))
#                 feats1 = extractor.extract(image1.to(device))
#                 matches01 = matcher({"image0": feats0, "image1": feats1})
#                 feats0, feats1, matches01 = [
#                     rbd(x) for x in [feats0, feats1, matches01]
#                 ]  # remove batch dimension

#                 kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
#                 m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

#                 axes = viz2d.plot_images([image0, image1])
#                 viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
#                 viz2d.save_plot(f"./output/lightglue/disk/lung-lesion_{l}/scale-"+k+f"pc/matching/keypoints_{j}_1HE_"+i+".jpg", dpi=300)
#                 plt.close()
#                 kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
#                 viz2d.plot_images([image0, image1])
#                 viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
#                 viz2d.save_plot(f"./output/lightglue/superpoint/lung-lesion_{l}/scale-"+k+f"pc/visualizaiton/keypoints_{j}_1-HE_"+i+".jpg", dpi=300)
#                 plt.close()

# stack = ["2-cd31","3-Pro-SPC", "4-cc10", "6-ki67"]
# stack2 = ['50']
# stack3 = [1024]

# for j in stack3:
#     extractor = SuperPoint(max_num_keypoints=j).eval().to(device)  # load the extractor
#     for k in stack2:
#         for i in stack:
#             if k == '100':
#                 image_0 = "lung-lobes_1/scale-"+k+"pc/29-039-U-35W-Izd1-1-HE.png"
#                 image_1 = "lung-lobes_1/scale-"+k+"pc/29-039-U-35W-Izd1-"+i+".png"
#             else:
#                 image_0 = "lung-lobes_1/scale-"+k+"pc/29-039-U-35W-Izd1-1-HE.jpg"
#                 image_1 = "lung-lobes_1/scale-"+k+"pc/29-039-U-35W-Izd1-"+i+".jpg"
#             image0 = load_image(image_0)
#             image1 = load_image(image_1)

#             feats0 = extractor.extract(image0.to(device))
#             feats1 = extractor.extract(image1.to(device))
#             matches01 = matcher({"image0": feats0, "image1": feats1})
#             feats0, feats1, matches01 = [
#                 rbd(x) for x in [feats0, feats1, matches01]
#             ]  # remove batch dimension

#             kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
#             m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

#             axes = viz2d.plot_images([image0, image1])
#             viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
#             # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
#             viz2d.save_plot("./output/lightglue/superpoint/lung-lobes_1/scale-"+k+"pc/matching/keypoints_"+f'{j}'+"_1-HE_"+i+".png", dpi=300)
#             kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
#             plt.close()
#             viz2d.plot_images([image0, image1])
#             viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
#             viz2d.save_plot("./output/lightglue/superpoint/lung-lobes_1/scale-"+k+"pc/visualization/keypoints_"+f'{j}'+"_1-HE_"+i+".png", dpi=300)
#             plt.close()

from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

# 추론 및 저장 루프
for l in range(1,3):
    output_dir = f"./output/lightglue/matching_result/lung-lesion_{l}/"
    for j in stack3:
        extractor = SuperPoint(max_num_keypoints=j).eval().to(device)  # 특징 추출기 초기화
        for k in stack2:
            for i in stack:
                # 이미지 로드
                image_0_path = f"lung-lesion_{l}/scale-{k}pc/29-041-Izd2-w35-He-les{l}.jpg"
                image_1_path = f"lung-lesion_{l}/scale-{k}pc/29-041-Izd2-w35-"+i+f"-les{l}.jpg"
                image0 = load_image(image_0_path)
                image1 = load_image(image_1_path)

                # 특징 추출 및 매칭
                feats0 = extractor.extract(image0.to(device))
                feats1 = extractor.extract(image1.to(device))
                matches01 = matcher({"image0": feats0, "image1": feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

                # 매칭된 keypoints 추출
                kpts0, kpts1, matches, scores = feats0["keypoints"], feats1["keypoints"], matches01["matches"], matches01["scores"]
                m_kpts0, m_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()


                # 원본 이미지 저장
                img0_save_path = Path(output_dir) / f"original_37-HE.jpg"
                img1_save_path = Path(output_dir) / f"original_{i}.jpg"

                pil_image0 = to_pil(image0.cpu())  # GPU에 있을 경우 CPU로 옮겨야 변환 가능
                pil_image1 = to_pil(image1.cpu())
                pil_image0.save(img0_save_path)
                pil_image1.save(img1_save_path)

                matches_df = pd.DataFrame(matches.cpu().numpy(), columns=["keypoint_idx_img0", "keypoint_idx_img1"])
                scores_df = pd.DataFrame(scores.cpu().numpy(), columns=["score"])
                keypoints_df0 = pd.DataFrame(m_kpts0, columns=["x_img0", "y_img0"])
                keypoints_df1 = pd.DataFrame(m_kpts1, columns=["x_img1", "y_img1"])

                # R, t = kabsch_algorithm(m_kpts0, m_kpts1)
                # rotation_matrix_df = pd.DataFrame(R, columns=["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"])
                # translation_vector_df = pd.DataFrame(t, columns=["tx", "ty", "tz"])

                scores_df = pd.DataFrame(scores.cpu().numpy(), columns=["score"])


                results_df = pd.concat([keypoints_df0, keypoints_df1, matches_df, scores_df], axis=1)
                output_matches_path = f"./output/lightglue/matching_result/lung-lesion_{l}/matches_scores_{j}_37-HE_{i}.csv"
                results_df.to_csv(output_matches_path, index=False)
                
                
# output_dir = f"./output/lightglue/matching_result/mammary-gland_1/"
# for j in stack3:
#     extractor = SuperPoint(max_num_keypoints=j).eval().to(device)  # 특징 추출기 초기화
#     for k in stack2:
#         for i in stack:
#             # 이미지 로드
#             image_0_path = f"mammary-gland_1/scale-{k}pc/s1_37-HE_A4926-4L.jpg"
#             image_1_path = f"mammary-gland_1/scale-{k}pc/s1_"+i+"_A4926-4L.jpg"
#             image0 = load_image(image_0_path)
#             image1 = load_image(image_1_path)

#             # 특징 추출 및 매칭
#             feats0 = extractor.extract(image0.to(device))
#             feats1 = extractor.extract(image1.to(device))
#             matches01 = matcher({"image0": feats0, "image1": feats1})
#             feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

#             # 매칭된 keypoints 추출
#             kpts0, kpts1, matches, scores = feats0["keypoints"], feats1["keypoints"], matches01["matches"], matches01["scores"]
#             m_kpts0, m_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()


#             # 원본 이미지 저장
#             img0_save_path = Path(output_dir) / f"original_37-HE.jpg"
#             img1_save_path = Path(output_dir) / f"original_{i}.jpg"

#             pil_image0 = to_pil(image0.cpu())  # GPU에 있을 경우 CPU로 옮겨야 변환 가능
#             pil_image1 = to_pil(image1.cpu())
#             pil_image0.save(img0_save_path)
#             pil_image1.save(img1_save_path)

#             matches_df = pd.DataFrame(matches.cpu().numpy(), columns=["keypoint_idx_img0", "keypoint_idx_img1"])
#             scores_df = pd.DataFrame(scores.cpu().numpy(), columns=["score"])
#             keypoints_df0 = pd.DataFrame(m_kpts0, columns=["x_img0", "y_img0"])
#             keypoints_df1 = pd.DataFrame(m_kpts1, columns=["x_img1", "y_img1"])

#             # R, t = kabsch_algorithm(m_kpts0, m_kpts1)
#             # rotation_matrix_df = pd.DataFrame(R, columns=["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"])
#             # translation_vector_df = pd.DataFrame(t, columns=["tx", "ty", "tz"])

#             scores_df = pd.DataFrame(scores.cpu().numpy(), columns=["score"])


#             results_df = pd.concat([keypoints_df0, keypoints_df1, matches_df, scores_df], axis=1)
#             output_matches_path = f"./output/lightglue/matching_result/mammary-gland_1/matches_scores_{j}_37-HE_{i}.csv"
#             results_df.to_csv(output_matches_path, index=False)