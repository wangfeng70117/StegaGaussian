# StegaGaussian: High-fidelity steganography for 3D Gaussian splatting based on frequency decomposition

# Introduction
Recent advances in 3D Gaussian Splatting (3DGS) have enabled high-quality scene reconstruction and real-time novel view synthesis. However, copyright protection in 3DGS remains largely unexplored. Existing watermarking methods typically rely on watermark neural framework and embed short bit sequences, which limits their capacity and restricts the expression of rich ownership information. In this study, we propose StegaGaussian, a novel steganography designed for 3DGS. It encodes the singular values as message extracted from the low-frequency subband of the watermark image into the corresponding low-frequency subband of the rendered images by subtly modifying the color of Gaussians, while using the remaining components of the watermark image as a private key for faithful reconstruction. Subsequently, to ensure resilience to viewpoint changes and perturbations, we extract the message from the low-frequency subband of rendered images using a dedicated watermark decoder network. The extracted message, together with the private key, are then used to reconstruct the watermark image. Extensive experiments demonstrate that our method achieves accurate watermark extraction, high-fidelity recovery, and robust performance under various perturbations \textcolor{red}{with the unique decoder trained for each scene}.  Quantitatively, StegaGaussian achieves over 50 dB PSNR in watermark reconstruction and maintains high rendering quality across large-scale and complex scenes, highlighting its practical applicability for 3DGS copyright protection.

# Pipeline
<img width="2148" height="760" alt="pipeline" src="https://github.com/user-attachments/assets/36bc7a05-cf42-4a79-a705-b0334ca86bc8" />


# Highlights
-  We introduce a key‐message decomposition strategy for watermark images by combining DWT and SVD. We use the singular values of the low-frequency subband as the embedding message, while the remaining elements serve as a private key for accurate watermark recovery.
- We design a CBAM‑enhanced decoder to robustly extract the message from low‑frequency subbands of rendered images, maintaining accuracy even under severe noise, compression, and viewpoint variations.

- We propose a negative supervision strategy that trains the decoder with both watermarked and clean inputs, ensuring that it produces meaningful messages only for watermarked renders.

- Experiments on diverse 3D scenes confirm that our method achieves strong imperceptibility, reliable watermark extraction, and robustness against common distortions.

```
python train_watermark_lambda.py -s DATA_PATH -m (3GDS OUTPUT_PATH) --water_iterations 40000 --J 1 --lambda_render 12 --lambda_wm 12 --lambda_w 1 --lambda_neg 15 
```

We have written all the scripts we run to the run_scripts.sh. You can run by 

```
python run_train.py
```
