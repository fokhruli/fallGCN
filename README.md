# fallGCN

## extract keypoints/landmarks from videos using mediapipe

We utilized medipipe module to extract keypoint frame by frame in real-time. So given videos of Nx1024x912, we get Nx33x3. 

mathmatically,

$$
\mathcal{L}_v^s = \textbf{F}(\mathcal{V_v^s})
$$

Here For each videos of each viewpoint ($v$) and scenario ($s$) $\mathcal{V} \in Nx1024x912$, the mediapipe pose estimation model, \textbf{F} provide landmark as given, $\mathcal{L} \in Nx33x3$.

Now using this landmark for all videos and create our desired dataset. For this take help from report given by website of dataset. The original dataset have 9 specific label, and we convert them into fall or non-fall classes by only taking ``Falling'' class in fall (=1) and non-fall(=0) otherwise. mathmatically, 
[give a preciwise function for this]


Table: orignial class mapping
1 Walking, standing up
2 Falling
3 Lying on the ground
4 Crounching
5 Moving down
6 Moving up
7 Sitting
8 Lying on a sofa
9 Moving horizontaly

And to sample the dataset, We take the samples as follows, for each window we have 90 frame and we take label which is after 60 frames, then we take next window from intial frame of previous window +15 and simillarly take label of frame's label of next 60 frame. For feature based data we convert given skelton data and convert them 45 dimentional vector for each frame. the details feature extraction process given in sections [will add] [rewrite this in more academic manner]

[make an algorithm]

Now our model is as follows,

we take input of 90x33x3 and 90x45 two input which process by two stream network. The first stream is based on spatio-temporal Graph neural network (STGCN) = BiLSTM which take 90x33x3 input and Graph adjacency and process by implementated model and provide 1x80 dimentional vector. the second stram, process 90x45 input and provide output of 80 dimentional vector. The second stream has 1 layer neural net and 1 BiLSTM layer. Then we fuse model by concatening and then use a FC layer to provide out probability. [provide a mathmatics]

the optimization is processed by weighted BCEloss which take consideration from imballance of total samples of fall/non-fall. so weight is calculated by factoring more in loss part of fall loss fraction which seen improvemt.

[a figure of whole model with whole pipeline, give loss function here]
