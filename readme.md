# GANs Project
---------------
By training 2 competing Neural Networks (one to generate fake images; one to detect fake images), you can build a top-notch image generation system. This is called **"GANs" - Generative Adversarial Network**.

##Summary
My GANs (and all GANs in general) uses 2 neural networks to **generate (from scratch) images of any target subject**. The 2 Neural Networks are as follows:
1. A **discriminator (D)** that we train to differentiate generated-images (from our network) and non-generated images (real) from elsewhere
2. A **generator (G)** that creates fake images, using the D's feedback to improve itself

##Specifications
- G takes a random 100-dimensional vector 
- applies sets of convolutions/normalizations/ReLu layers
- outputs a "fake image" (3-dimensional vector)

- D takes the G's fake-image & real-image (ground truth)
- outputs a "real-ness score" from 0-1 for G's image (0 ideally); and the real images (1 ideally)
- back-propogates its error & advises G on how to improve

