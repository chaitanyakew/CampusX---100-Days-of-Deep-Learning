# 100 Days of Deep Learning - Notes

My personal notes while following the *100 Days of Deep Learning* course. Only lectures that have notes are included.

## Contents

- [Day 2 - What is Deep Learning](#day-2---what-is-deep-learning)
- [Day 3 - Types of Neural Network](#day-3---types-of-neural-network)
- [Day 4 - What is Perceptron](#day-4---what-is-perceptron)
- [Day 5 - Perceptron Trick](#day-5---perceptron-trick)
- [Day 6 - Perceptron Loss Function](#day-6---perceptron-loss-function)
- [Day 8 - MLP Notation](#day-8---mlp-notation)
- [Day 9 - Multi layer Perceptron](#day-9---multi-layer-perceptron)
- [Day 10 - Forward Propagation](#day-10---forward-propagation)
- [Day 14 - Loss Functions](#day-14---loss-functions)
- [Day 18 - Vanishing Gradient problem](#day-18---vanishing-gradient-problem)
- [Day 19 - MLP Memoization](#day-19---mlp-memoization)
- [Day 20 - Gradient Descent in Neural Networks](#day-20---gradient-descent-in-neural-networks)
- [Day 21 - Improve Performance of Neural Network](#day-21---improve-performance-of-neural-network)
- [Day 22 - Early Stopping](#day-22---early-stopping)
- [Day 23 - Normalizing inputs - Data Scaling in Neural Network](#day-23---normalizing-inputs---data-scaling-in-neural-network)
- [Day 24 - Dropout Layer in Deep Learning](#day-24---dropout-layer-in-deep-learning)
- [Day 26 - Regularization](#day-26---regularization)
- [Day 27 - Activation Functions](#day-27---activation-functions)
- [Day 28 - ReLU Variants](#day-28---relu-variants)
- [Day 29 - Weight Initialization Techniques](#day-29---weight-initialization-techniques)
- [Day 30 - Xavier/Glorot & He Weight Initialization Techniques](#day-30---xavierglorot--he-weight-initialization-techniques)
- [Day 31 - Batch Normalization](#day-31---batch-normalization)
- [Day 32 - Optimizers in Deep Learning](#day-32---optimizers-in-deep-learning)
- [Day 33 - Exponentially Weighted Moving Average or Exponentially Weighted Average](#day-33---exponentially-weighted-moving-average-or-exponentially-weighted-average)
- [Day 34 - SGD with Momentum](#day-34---sgd-with-momentum)
- [Day 35 - NAG - Nesterov Accelerated Gradient](#day-35---nag---nesterov-accelerated-gradient)
- [Day 40 - CNNs](#day-40---cnns)
- [Day 41 - CNN vs Visual Cortex](#day-41---cnn-vs-visual-cortex)
- [Day 42- Convolution Operation](#day-42--convolution-operation)
- [Erai CNN Backpropagation](#erai-cnn-backpropagation)
- [Day 43 - Padding and Strides](#day-43---padding-and-strides)
- [Day 44 - Pooling Operation](#day-44---pooling-operation)


## Day 2 - What is Deep Learning

CNN is good for image, video tasks because they can automatically learn spatial features such as edges, textures, and shapes.  
RNN is good for speech (Today Transformers have largely replaced LSTMs and GRUs)  
GAN is good for generation images, videos, text (Modern text generation is dominated by Transformer-based models such as GPT.)

Deep Learning algorithms use multiple layers to progressively extract higher level features from raw input.

In machine learning (ML), feature learning or representation learning is a set of techniques that allow a system to automatically discover the representations needed for feature detection or classification from raw data. This replaces manual feature engineering and allows a machine to both learn the features and use them to perform a specific task.

In ML we have to provide features while in DL, the algorithm automatically extracts features from the datasets. Features being -> size, color for a dog cat classifier. DL doesn't need manual feature engineering like ML. Lower layers may identify edges, while higher layers may identify complex features.

DL is data hungry than ML, DL performance increases as we provide more data. No manual feature selection for DL. Most ML algos are highly interpretable, hence high explainability. DL acts as a black box.

DL exploded later on as computation got cheaper, availability of labelled datasets increased, libraries like pytorch, tensorflow were made for easy DL development (keras was intermediary library (High level API) used to talk with tensorflow) (pytorch was merged with caffe 2 for easier model deployment), pretrained models were made available like RESNET, YOLO, BERT, etc.


## Day 3 - Types of Neural Network

Types of NN -  
1. **MLP** - Multi Layer perceptron, commonly used for machine learning problems like regression or classification  
2. **CNN** - Convolutional Neural Network - contains at least one convolutional layer  
3. **RNN** - Recurrent Neural Network (LSTMs, GRUs)  
4. **Auto Encoders** - Autoencoders learn compressed representations of data and are commonly used for dimensionality reduction, denoising, and anomaly detection.  
5. **GAN** - Generative Adversarial Networks (GANs consist of a Generator and a Discriminator competing against each other.)

Perceptron can not work on non linear data, it can not learn XOR function at all.


## Day 4 - What is Perceptron

Perceptron works only on linearly separable data.  
More from Machine Leanring lecture - Perceptron


## Day 5 - Perceptron Trick

Trick perceptron - We don't use loss function here.  
Iterate over no of epochs  
- Just select point randomly.  
- If point is correctly classified do nothing.  
- If point is not correctly classified, update weights with weights updation rule.


## Day 6 - Perceptron Loss Function

Problems with Perceptron Trick -  
- Cant quantify results  
- May or may not converge as the random points that we are choosing are not good for updation.


## Day 8 - MLP Notation

Multi-Layer Perceptron

Data =  m * n  
m = no of rows  
n = no of attributes  
Each row is denoted by Xi  
X<sub>100,4</sub> would be 4th column for 100th row

For biases, b<sub>ij</sub> would be bias in i<sup>th</sup> hidden layer and j<sup>th</sup> node  
b<sub>1,2</sub> would be bias for node in 1<sup>st</sup> layer and 2<sup>nd</sup> node


## Day 9 - Multi layer Perceptron

MLP can generalize on linearly inseparable data, unlike perceptron.

Activation functions like **sigmoid**, **ReLU**, and **tanh** introduce **non-linearity**, allowing neural networks to learn: images, speech, language and complex patterns

Step function - The original perceptron was designed for **binary classification**.  
Sigmoid function - This squashes the input in range of 0 to 1. The output can be interpreted as probabilities op 0.95 - 95% confidence of the classification.

Why is sigmoid preferred over the step function in neural networks?  
Sigmoid is differentiable where step function is not.  
The sigmoid function is: smooth, continuous and differentiable


## Day 10 - Forward Propagation

The matrix **_A_** has three rows and two columns. More generally, if the matrix has _m_ rows and _n_ columns and contains real values, you can characterize it with the following notation:  
A[3,2] 3 rows, 2 columns, a 3 by 2 matrix  
For dot product m,n matrix . n,1 vector = m,1 vector  
[Source](https://www.kdnuggets.com/2021/02/essential-math-data-science-matrices-matrix-product.html)  
Matrix Vector Dot product  
![Pasted image 20260701132115.png](Attachments/Pasted%20image%2020260701132115.png)

Matrix Matrix Dot product  
m,n matrix . n,p matrix = m,p matrix  
![Pasted image 20260701132216.png](Attachments/Pasted%20image%2020260701132216.png)

In forward propagation, each neuron computes the weighted sum of its inputs by multiplying each input by its corresponding weight and adding a bias term.  
This weighted sum is then passed through an activation function, such as ReLU, sigmoid, or tanh. The output of the activation function is called the **activation** of the neuron and serves as the input to the neurons in the next layer. This process is repeated for every neuron in each hidden layer until the output layer is reached.  
For **binary classification**, the output layer typically consists of a single neuron with a **sigmoid** activation function, which produces a probability between 0 and 1.  
For **multi-class classification**, the output layer usually contains one neuron for each class and uses the **softmax** activation function to produce a probability distribution over all classes. The class with the highest probability is taken as the predicted class.


## Day 14 - Loss Functions

Loss Function of evaluating how well is your algorithm is modelling the dataset.  
High loss = poor modelling  
Low loss = great modelling

Why loss functions are important - because you can't improve something that you cant measure and we want to minimize the loss.

Types of loss functions -  
- Regression - Mean squared error, mean absolute error, Huber loss  
- Classification - binary cross entropy, categorical cross entropy, hinge loss  
- Autoencoders - KL Divergence  
- GAN - minmax, gan loss, discriminator loss  
- Object Detection - focal loss  
- Embeddings - triplet loss

Each has its own advantages and disadvantages.

Loss function - calculated for a single training example  
Cost function - calculated for whole dataset

1. Mean Squared Error (Squared Loss, L2 Loss) -  
	- (y<sub>i</sub>-Y<sub>i</sub>)<sup>2</sup> CF= 1/n summation (y<sub>i</sub>-Y<sub>i</sub>)<sup>2</sup> - squared to make everything positive, it becomes a quadratic function, punishes farther points more harshly.  
	- Advantages - easy to interpret, always differentiable, makes it easier for GD, 1 local minima  
	- Disadvantages - error unit is squared, not robust to outliers, last neurons activation function should be linear only.

2. Mean Absolute Error (L1 Loss)-  
	- |y<sub>i</sub>-Y<sub>i</sub>| CF = 1/n summation |y<sub>i</sub>-Y<sub>i</sub>|  
	- Advantages - easy to understand, unit stays the same for loss, robust to outliers as it not punishes farther points as harshly.  
	- Disadvantages - not differentiable, GD can not be applied directly, we have to calculate subgradients, becomes computationally expensive.

3. Huber Loss - Way more outliers, MAE MSE dont work that good.

4. Binary Cross Entropy (Log Loss)-  
	- Used in binary classification  
	-  -y log(Y) - (1-y)log(1-Y)  
	-  -1/n summation { y log(Y) - (1-y)log(1-Y) }  
	- last neurons activation function should be sigmoid only.  
	- Advantage - differentiable  
	- Disadvantage - multiple local minima

5. Categorical Cross Entropy -  
	- Used in Multi Class Classification  
	- Activation function should be softmax  
	- Have to one hot encode predicted class

6. Sparse Categorical Cross Entropy -  
	- Don't have to do one hot encoding  
	- Faster than CCE


## Day 18 - Vanishing Gradient problem

Prominently seen when sigmoid or tanh activation functions are used.  
Neural networks are trained using backpropagation, where gradients are computed using the chain rule. If many of these derivatives are less than 1, multiplying them repeatedly causes the gradient to shrink toward zero.  
When gradients vanish:  
- Early layers learn very slowly.  
- Training becomes extremely slow.  
- The network may fail to learn useful features.  
- Deep networks perform poorly despite having many layers.

How do we solve it?  
- ReLU and its variants (Leaky ReLU, ELU, GELU) (helps reduce the problem because its derivative is 1 for positive inputs)  
- Proper weight initialization (e.g., Xavier/Glorot or He initialization)  
- Batch Normalization, which stabilizes activations and gradients  
- Residual connections (skip connections), as used in ResNets, which let gradients flow more easily through the network  
- Specialized architectures like **LSTM** and GRU for recurrent neural networks

How to recognize -  
- Loss doesnt change much after each epoch  
- Plot weights graph and check is it changing or not

How to solve -  
- Reduce model complexity (Reduce number of hidden layers). Less no of derivatives to multiply so less chance that vanishing gradient happens. But this does mean that model will not learn complex patterns.  
- Using ReLU activation function. Introduces dying ReLU problem. (Derivative becomes 0, so whole update is 0 hence dying.)  
- Proper weights initialization techniques e.g. Xavier/Glorot  
- Using Batch normalization  
- Using residual network

Exploding Gradient Problem - Seen in Recurrent Neural Network  
Derivatives become bigger than 1, hence their multiplication becomes way more, hence exploding gradient. We use gradient clipping to mitigate this.


## Day 19 - MLP Memoization

Memoization - optimization technique to speed up computer programs by storing the results of expensive function calls and returning the cached result when the same inputs occur again.  
You trade in storage for reducing time complexity. 


## Day 20 - Gradient Descent in Neural Networks

1. Batch Gradient Descent - (Vanilla GD)  
	Prediction are done over entire dataset and gradient (derivative of loss wrt weight) is calculated and weights are updated once per epoch.  
	50 epoch - 50 updates  
	Stable, smooth convergence but slow for large datasets, high memory usage.
	
2. Stochastic Gradient Descent -  
	Weights are updated for each row in the dataset, this is done for each epoch  
	100 datapoints, 50 epoch = 100 * 50 = 5000 updates  
	Faster updates, works well for large datasets, can escape local minima but noisy updates, less stable convergence.

3. Mini Batch Gradient Descent -  
	Here we divide the dataset into small batches like 32,64 etc. and perform BGD for that batch, iterate to cover all batches.  
	We make prediction for that batch, compute gradient and then update weights.  
	50 epoch, 100 datapoints (4 batches) = 50 * 4 = 200 updates

Batch size is in multiples of 2 -  
	Powers-of-two batch sizes often make better use of GPU hardware because they align well with the GPU's parallel execution units and memory architecture, leading to higher utilization and, in some cases, faster computation.


## Day 21 - Improve Performance of Neural Network

1. Finetuning the hyperparameters-  
	1. Choosing No of hidden layers  
		- No fixed rule for no of hidden layers  
		- It is often better to use multiple hidden layers with a reasonable number of neurons rather than a single extremely wide hidden layer, but this depends on the problem.  
		- Adding too many hidden layers can lead to - Overfitting Increased training time Vanishing/exploding gradients.  
		- The optimal number of hidden layers is usually determined through experimentation and validation performance  
		- These Deep neural networks can be used in transfer learning.  
	2. Choosing No of neurons per layer  
		- Input layer - no of attributes  
		- Output layer -  
			- Binary classification → 1 neuron  
			- Multi-class classification → Number of classes  
			- Regression → 1 neuron (or more if predicting multiple values)  
		- Hidden layer -  
			- There is no universally correct number of neurons.  
			- Previously - Follow pyramid structure 64, 32, 16. The idea was to gradually compress information. Cause early layers learn primitive patterns and later layers learn complex patters, there will be a lot no of primitive patters so higher no of neurons. This is not affect performance that much you can do it anyhow like 32, 32, 32.  
			- But it should be more than what is required because when we add less no of neurons in the layers and that layer doesn't learn the patterns, later layers cannot recover information that was never learned in the first place.  
	3. Choosing Learning rate  
	4. Choosing Optimizer  
	5. Choosing batch size -  
		- Batch  
		- Stochastic  
		- Mini Batch -  
			- Here batch size is the hyperparameter.  
			- If we take it smaller like 8-32, training is slower, Often leads to better generalization, Training can take longer overall because the GPU is not utilized as efficiently.  
			- If we take it bigger like 8192, training is faster because of better GPU utilization. But this requires much more GPU memory, can use learning rate scheduler. Warm-up means increasing the learning rate only during the initial few epochs, not throughout training. After warm-up, the learning rate is usually held steady for a while and then decreased according to a scheduler.  
	6. Choosing activation function  
	7. Choosing No of epochs - use early stopping.  
2. By solving problems like -  
	1. Vanishing Gradient -  
		- Can use weight initialization techniques  
		- Change activation functions  
		- Use batch normalization  
	2. Exploding Gradient -  
		- Gradient Clipping  
	3. Not enough data  
		- Transfer learning  
		- Unsupervised pretraining  
		- Adding more data (Data Augmentation (Random flips to images))  
	4. Slow training (can it be made faster)  
		- Using optimizers  
		- Using learning rate scheduler  
	5. Overfitting  
		- L1 and L2 regularization  
		- Early Stopping  
		- Dropout Method


## Day 22 - Early Stopping

Implemented using callback in keras  
param -  
- monitor - val_loss  
- min_delta - minimum change in monitored quality to qualify as an improvement, if change is less than this then that epoch would be counted as no improvement  
- patience - no of epochs with no improvement, after reaching patience training will stop.  
- verbose - 0 is silent, 1 displays msg  
- mode - auto min max, min - monitored quantity stopped decreasing, max - monitored quantity has stopped increasing, auto - direction is inferred from the name of monitored quality  
- baseline - usually none  
- restore best weights - restore those weights where the model had best value for monitored quality


## Day 23 - Normalizing inputs - Data Scaling in Neural Network

When one feature is large than other features then it dominates the weight update. The network struggles to learn appropriate weights because the scales are so different. As the gradient is dependent on the inputs, the gradient becomes very large than other. The gradient descent keep oscillating instead of moving directly to minimum.  
This causes: Very large updates for one weight, very small updates for another, slow and unstable convergence.  
normalization leads to more stable training and often better model performance.  
Some normalization methods-  
1. Min-Max Normalization (Min-Max Scaling) -  
	- This is done when the min and max values are known like for cgpa, for salary we don't know what is min and max  
	- x' = (x - min) / (max - min)  
2. Standardization (Z-score Normalization)  
	- x' = (x - mean) / standard_deviation  
	-  x' = (x - μ) / σ  
3. Mean Normalization  
	- x' = (x - mean) / (max - min)  
4. Max Absolute Scaling  
	- x' = x / max(|x|)  
5. Robust Scaling  
	- x' = (x - median) / IQR  
6. Unit Vector Normalization (L1/L2 Normalization)  
	- x' = x / ||x||  
	- For a vector x1, x2, ..., xn :  
	- |x|| = sqrt(x1² + x2² + ... + xn²)


## Day 24 - Dropout Layer in Deep Learning

A Dropout layer is a regularization technique used to reduce overfitting in neural networks.  
The main idea is simple: During training, randomly "drop" (turn off) some neurons so that the network cannot become too dependent on any particular neuron.  
Dropout rate - chosen for each layer - Probability of dropping a neuron - 0.2 - 20% of neurons are dropped.

Overfitting - increase p, underfitting - decrease p

Advantages -  
- Reduces overfitting  
- Prevents neurons from becoming too dependent on each other

Disadvantages -  
- Training becomes slower  
- Can lead to underfitting


## Day 26 - Regularization

Used to reduce overfitting.  
L1, L2, L1 + L2 Regularization are weight-penalty methods.  
L2 is more prominently used.

##### L1 Regularization (Lasso)

![Pasted image 20260708122937.png](Attachments/Pasted%20image%2020260708122937.png)  
Penalizes the absolute value of weights, can push weights to zero.  
Used when subset of weights actually matter.  
Gives sparse model with multiple 0 weights.

##### L2 Regularization (Ridge / Weight Decay)

![Pasted image 20260708122745.png](Attachments/Pasted%20image%2020260708122745.png)  
Pushes weights toward small values but rarely to zero. It punishes large weights much more than small ones. This is often applied directly in the optimizer as weight decay.  
Used when most weights are somewhat relevant but large

###### L1 + L2 Combined (Elastic Net)

![Pasted image 20260708123554.png](Attachments/Pasted%20image%2020260708123554.png)  
best of both - some weights get zeroed out, and they become smaller


## Day 27 - Activation Functions

Neurons calculate weighted sum, pass that weighted sum into a function. That function is used to introduced non-linearity. That is called activation function. Without this, the model behaves like a linear model and can not capture non-linear data.  
E.g. Sigmoid, tanh, ReLU, softmax, leaky ReLU, etc  
Ideal activation function should be  
- non-linear  
- differentiable (to utilize gradient descent)  
- computationally inexpensive  
- zero centered - function output is both negative and positive  
- Non-saturating - saturating squeezes the input in a range like sigmoid or tanh, this introduced vanishing gradient problem. Non-saturating doesn't squeeze the input.

1. Sigmoid Function -  
	![Pasted image 20260708125327.png](Attachments/Pasted%20image%2020260708125327.png)  
	Now almost never used in hidden layers because of disadvantages below. Used in output layer for binary classification.
	
	Advantages -  
	- squeezes output in range 0-1, can treat this as probability.  
	- non-linear  
	- differentiable

	Disadvantages -  
	- saturating function - introduced vanishing gradient function.  
	- Non zero-centered - always outputs either negative or positive values, causes training to become slow.  
		- For updating weights the formula is new weight = weight - lr * delta * input (activation from previous layer).  
		- Now delta here is the error signal computed during backpropagation which tells the neuron how much changing this neuron's output would affect the final loss.  
		- Activation function which is non zero-centered will always output all positive or all negative values. sigmoid will always have positive values, ReLU has 0 and positive values.  
		- The delta is constant for a neuron during backpropagation. When weight updation rule is applied the all the weights either decrease or increase.  
		- This prevents some weights to increase and some to decrease within that update, hence increasing the training time.  
	- It is computationally expensive.

2. Tanh Function -  
	![Pasted image 20260708134819.png](Attachments/Pasted%20image%2020260708134819.png)  
	Tanh solved the zero-centered problem of sigmoid function but the vanishing gradient problem because of squeezing is still present.

	Advantages -  
	- Non-linear  
	- Differentiable  
	- Zero-centered

	Disadvantages -  
	- Saturating function - squeezes input  
	- Computationally expensive

3. ReLU - Rectified Linear Unit  
	![Pasted image 20260708135110.png](Attachments/Pasted%20image%2020260708135110.png)
	
	Advantages -  
	- Non-linear  
	- Non saturated in positive region (no vanishing gradient)  
	- Computationally inexpensive

	Disadvantages -  
	- Not differentiable at 0  
	- Not zero-centered (batch normalization is used to overcome this)


## Day 28 - ReLU Variants

Rectified Linear Unit

Dying ReLU Problem -  
When a neuron dies, i.e. it gives output 0 for any input, it stays dead forever. ReLU = max(0, x)  
When weighted sum for a particular neuron becomes 0 or less than 0 then ReLU's output also becomes zero for almost all inputs. That makes no change to the weight and the weight doesn't change. hence the dying ReLU problem.

High learning rate - When the learning rate is high, it turns the weight negative and negative weight leads to weighted sum to be negative which makes the output of ReLU 0, and we again face dying ReLU problem.

High negative bias - makes weighted sum negative -> dying ReLU

Solving Dying ReLU Problem -  
1. Setting low learning rate  
2. Proper weight initialization  
3. Setting bias a positive value  
4. Use other variants of ReLU

Variants of ReLU -  
Linear Variants - Leaky ReLU, Parametric ReLU  
Non-Linear Variants - Exponential linear Unit - ELU, Scaled exponential linear Unit - SELU 

1. Leaky ReLU -  
	![Pasted image 20260709001505.png](Attachments/Pasted%20image%2020260709001505.png)  
	α is a small fixed constant like 0.01  
	For z < 0, the derivative out have become 0 but now it is 0.01, which makes small weight updates at least.
	
	Advantages -  
	- Non-saturated  
	- Computationally inexpensive  
	- No dying ReLU problem  
	- closer to zero center

	Disadvantage -  
	- α is a fixed hyperparameter, it may not be optimal for every dataset

2. Parametric ReLU - PReLU  
	![Pasted image 20260709001505.png](Attachments/Pasted%20image%2020260709001505.png)  
	α is trainable hyperparameter which is dependable on data, needs to be learned

3. ELU - 

	![Pasted image 20260709002056.png](Attachments/Pasted%20image%2020260709002056.png)  
	α is constant  
	Performance is better than ReLU

	Advantages -  
	- Non zero derivative  
	- Closer to zero centered  
	- Better generalization  
	- No dying ReLU problem  
	- Always continuous  
	- Differentiable

	Disadvantages -  
	- More computationally expensive than ReLU, Leaky ReLU  
	- Saturates for large negative z, so vanishing gradient can still occur

4. SELU - 

	![Pasted image 20260709002302.png](Attachments/Pasted%20image%2020260709002302.png)

	where λ≈1.0507 and α≈1.6733

	It is self normalizing - faster convergence

	Disadvantages -  
	- Self-normalization only works under specific conditions: requires **LeCun normal initialization** and a plain feedforward architecture.  
	- Works less reliably with CNNs, ResNets/skip-connections, or in combination with standard Dropout (needs a special variant called AlphaDropout instead).


## Day 29 - Weight Initialization Techniques

Why it is important -  
	If initial values are wrong then we might face problems like vanishing gradient, exploding gradient, slow convergence, etc.

Wrong weight initialization techniques -  
- Zero initialization -  
	- If weights are initialized as zeros, then every neuron will receive the same pre-activation value z=0. Hence the output from the activation function is all the same. So the neurons in the next hidden layer will always receive the same input as other neurons. And during backpropagation, they all receive the same gradient as well. As a result, the neurons will update identically and the weights coming out from that neuron will have the same value and so does the other neurons in the same layer. hence, the hidden layer will act as a single neuron, this is a symmetry-breaking failure and will never capture non-linearity.  
	- If weights are initialized as zero, the weights will never change and remain zero while using ReLU.  
	- For sigmoid and tanh, the weights will get some value but they remain the same for all neurons in the layer.

- Initialization with constant non-zero value -  
	- If weights are initialized with a constant non-zero values then every neuron will receive the same pre-activation value and identical gradient during backpropagation. This results in similar situation as zero initialization, the same symmetry collapse, with all the weights from a neuron becoming similar and the whole layer acting as a single neuron and the model will never capture non-linearity.

- Random initialization -  
	- If we initialize weights with small random values, the value of z will be very small as the inputs are normalized. So for activation functions like tanh or sigmoid there is possibility of facing vanishing gradient problem.  
	- If we initialize weights with large random values, z becomes bigger and tanh and sigmoid face saturation and face vanishing gradient problem. ReLU suffers from unstable gradient and keeps zig zag pattern while reaching minima.


## Day 30 - Xavier/Glorot & He Weight Initialization Techniques

Xavier/Glorat Initialization (For Tanh, sigmoid , not for ReLU)-  
	n<sub>in</sub>​ = number of input neurons (fan-in)  
	n<sub>out​</sub> = number of output neurons (fan-out)  
	Variance = 1/n<sub>in​</sub>  
	std dev = sqrt(1/n<sub>in​</sub>)

-  Uniform Xavier Initialization

	 ![Pasted image 20260710160614.png](Attachments/Pasted%20image%2020260710160614.png)  
	U(a,b) - a uniform distribution between a and b
	
- Normal Xavier Initialization

	![Pasted image 20260710160712.png](Attachments/Pasted%20image%2020260710160712.png)
	


He Initialization (For ReLU)-  
	n<sub>in</sub>​ = number of input neurons (fan-in)  
- Uniform He Initialization  
	![Pasted image 20260710161758.png](Attachments/Pasted%20image%2020260710161758.png)
	
-  Normal He Initialization - 
	
	![Pasted image 20260710161726.png](Attachments/Pasted%20image%2020260710161726.png)	


## Day 31 - Batch Normalization

What is Batch Norm ?  
	Batch normalization is a technique used to normalize the outputs of a layer (activation values) during training. It helps neural networks train faster, more stable and often achieve better accuracy.  
	This normalization step is applied right before or after the non-linear function.  
	Normalization - making its mean 0 and standard deviation 1.

Covariate shift -  
	Covariate shift occurs when the distribution of the input features changes, but the relationship between the inputs and outputs remains the same.  
	When this happens, the trained model is not highly accurate on the test data.

Internal Covariate Shift -  
	ICS means the change in the distribution of network activations due to the change in network parameters during training.

How does it work -  
	2 ways -  
		1. Normalize the output from activation function  
		2.  Normalize the weighted sum value before putting it into activation function  
	When we send the data into batches (size m), using the activation values for all those m datapoints for each neuron, we calculate mean and standard deviation.  
	![Pasted image 20260729105627.png](Attachments/Pasted%20image%2020260729105627.png) ![Pasted image 20260729105640.png](Attachments/Pasted%20image%2020260729105640.png)  
	using that we normalize the activation value.  
	![Pasted image 20260729105613.png](Attachments/Pasted%20image%2020260729105613.png)  
	Then we Scale and shift using two learnable parameters - for scaling $\gamma$   and for shifting $\beta$. This is necessary as it enables the network to correctly represent the preferred distribution.  
	So the final equation becomes -  
	![Pasted image 20260729110149.png](Attachments/Pasted%20image%2020260729110149.png)  
	This works for testing as we have a batch of training data but for testing we only have a datapoint to be tested for. So we use exponentially weighted moving average EWMA.  
	![Pasted image 20260729111430.png](Attachments/Pasted%20image%2020260729111430.png) ![Pasted image 20260729111439.png](Attachments/Pasted%20image%2020260729111439.png)  
	where $\mu_\beta$ = mean and $\sigma^2_\beta$ is standard deviation of current batch. This $\beta$ is totally different from  
 one used while calculating mean and std dev.	

Advantages -  
- Makes training more stable (Hyperparameter values can be set in a wide range).  
- Training becomes faster. (Can use higher learning rate)  
- Acts as an regularizer.  
- Reduces impact of weight initialization technique.


## Day 32 - Optimizers in Deep Learning

We have seen how to improve performance of NN [Improve Performance of Neural Network](#day-21---improve-performance-of-neural-network), and mainly how to speed up the training of NN. This can be done by using [weight initialization](#day-29---weight-initialization-techniques) , [batch normalization](#day-31---batch-normalization) , [activation functions](#day-27---activation-functions) and now we will see - Optimizers.

Deep learning is an optimization problem where we have to find out the best possible combination of weights and biases such that the loss is minimized. For this we need an optimizer, We have been using [Gradient Descent](#day-20---gradient-descent-in-neural-networks) for this till now. 

###### Why do we need other optimizers | Challenges faced by Optimizers-
1. Choosing the correct value of learning rate -  
	Learning rate scheduling - Lr is reduced by using a predefined schedule. It does not depend on the dataset so, it is not necessary one schedule that works for one dataset might work for other dataset.  
2.  In gradient descent, we can not choose the learning rate for each weight, if one weight required to be reduced by 0.5 and another by 0.1, then this operation can not be performed as we only get to choose only one Lr. This is a restriction of Gradient Descent, that we get to move in only one direction and with only one magnitude.  
3. Getting stuck in local minima is a great challenge for gradient descent.  
4. Saddle Point - Saddle point is a point where slope in each direction is the same. This makes the $\dfrac{\delta L}{\delta w1}$ zero so the new weights that is supposed to be updated gets stuck.  
	<img src="Attachments/Pasted%20image%2020260715124859.png" width="293">


## Day 33 - Exponentially Weighted Moving Average or Exponentially Weighted Average

Exponentially Weighted Moving Average (EWMA) - Technique to find trends in time series data.  
The data point which came later into time will have more weightage as it is recent and weightage for older points will be reduced as we move forward.  
e.g. t=5 will be weighted more than t=1

EWMA = V<sub>t</sub> = $\beta$ V<sub>t-1</sub> + (1- $\beta$ ) $\theta$<sub>t</sub>  
	V<sub>t</sub> = EWMV at time t  
	$\beta$ = constant ranging from 0 to 1  
	$\theta$<sub>t</sub> = current data

V<sub>0</sub> is taken as 0 or $\theta$<sub>0</sub> i.e. first data point

if $\beta$ is 0.9 then it acts as normal average of $\dfrac{1}{1-\beta}$ i.e. 1/0.1 = 10 so it will act as normal average of previous 10 datapoints.  
High $\beta$ means more weightage to past datapoints, low $\beta$ means more weightage to current datapoints.  
Low beta tends to have line that is changing constantly, whereas high $\beta$ tends to produce a stable line.


## Day 34 - SGD with Momentum

[Linked Resource - Convex vs Non-Convex Function](https://www.youtube.com/watch?v=TXVtbgaEyms&t=1s)  
Convex function - A function is called convex if the line segment between any two distinct points on the graph of the function lies above or on the graph of the function between the two points. A strictly convex function has no more than one minimum. Non-Convex function can have more than one minima.  
Problems with Non-Convex Loss Function -  
- Getting stuck into local minima due to noisy gradient  
- Saddle point  
- High curvature  
- Consistent Gradient - slope changes slowly so training becomes slower

Vanilla GD or SGD can not navigate these problems. It makes either the training slower or the results is suboptimal solution. So we have to use optimization techniques like momentum.

###### Momentum - 
If 4 people tell us that the destination is ahead, we will drive faster. But only 2 people tell that it is further and 2 people tell that it has already passed, we will drive cautiously until confirmation. Same intuition is used while moving towards minima.  
This is why momentum increases training speed.

So the weight update rule $w_i^* = w_i - \alpha \times \frac{\partial L}{\partial w_i}$ in this context $w_{t+1} = w_t - \eta \times \nabla w_t$  
becomes -> $w_{t+1} = w_t - v_t$  
where $v_t = \beta v_{t-1} + \eta \nabla w_t$ 

<img src="Attachments/Pasted%20image%2020260906163907.png" width="423">

Effects of Beta -  
When beta is 0, SGD with momentum will act as normal SGD  
If beta is 1, loss will just keep oscillating and might not reach minima

Momentum increases training speed and it can help escape local minima.  
Sometimes, it can also overshoot and escape the global minima.


## Day 35 - NAG - Nesterov Accelerated Gradient

As we have seen for momentum, the weight update rule is -  
$w_{t+1} = w_t - v_t$  where $v_t = \beta v_{t-1} + \eta \nabla w_t$  
$w_{t+1} = w_t - ( \beta v_{t-1} + \eta \nabla w_t )$  
Here two terms are affecting the new weight, past velocity ($\beta v_{t-1}$) and gradient at that point ($\eta \nabla w_t$)

In NAG, we first look ahead by moving the weights in the direction of the previous momentum (First jump), then calculate the gradient at this look-ahead position. This gradient is used together with the momentum term to update the velocity, which is then used to update the weights (Second jump).  
- First Jump -  
	$w_t^{\text{lookahead}} = w_t - \beta v_{t-1}$ 

- Calculate Gradient at lookahead position -  
	$\nabla w_t^{\text{lookahead}}$

- Update Velocity -  
	$v_t = \beta v_{t-1} + \eta \nabla w_t^{\text{lookahead}}$

- Second Jump  
	$w_{t+1} = w_t - v_t$ 
 


## Day 40 - CNNs

Convolutional Neural Networks are a special kind of neural network for processing data that has a known grid like topology like time series data (1D) or images (2D)

Convolution layer performs convolution operation. It is used to extract features from images. One layer can extract primitive features, we combine those features and pass it on to another convolutional layer where it learn to detect complex patterns and so on. That's why we have multiple convolutional layers .

3 types of layers  
Convolution Layer  
Pooling Layer  
Fully Connected layer

Why not use ANN for image data ?  
1. High Computation Cost - 1920 * 1080 * 3 channels -> lots of weights in first layer only, takes lot of time to train the model  
2. Overfitting  
3. Loss of important info like spatial arrangement


## Day 41 - CNN vs Visual Cortex

The conclusion of the experiment on the cat was - we have two cells simple cell and complex cell.  
Simple cell is called orientation cell or feature detector cell, it has smaller receptive field. They perform edge detection.  
Complex cell detects complex features using info from simple cells and has bigger receptive field


## Day 42- Convolution Operation

*Read below chapter -ERAI CNN Backpropagation first.*

We initialize filters with some random value and with backpropagation the actual fiter values are learnt.

Feature map shape is defined by the input image and filter's shape  
If the image is n * n and the filter is m * m then the shape of the feature map is (n-m+1, n-m+1)  
for 28 * 28 and filter 3 * 3 we get 28-3+1 = 26 i.e. the shape of the feature map is 26 * 26

When there are multiple channels, filter is assumed to have the same no of channels. But, the result of convolution operation only has 1 channel.  
If the image is n * n * c and the filter is m * m * c then the shape of the feature map is (n-m+1, n-m+1)

Multiple filters -  
When we perform convolution operation on image with multiple filters, the result will be multiple feature maps. We can treat them as a block i.e. stack them. This will act as input to the next layer.


## Erai CNN Backpropagation

![Pasted image 20260715133009.png](Attachments/Pasted%20image%2020260715133009.png)  
a1-a25, This is our picture, 5x5 image

w1-w9, This is our kernel (filter) , it has its own bias term

After the convolution operation we get our feature map - z1-z4

where z are activation values  
z1 = a1·w1 + a2·w2 + a3·w3 + a6·w4 + a7·w5 + a8·w6 + a11·w7 + a12·w8 + a13·w9  
z2 = a3·w1 + a4·w2 + a5·w3 + a8·w4 + a9·w5 + a10·w6 + a13·w7 + a14·w8 + a15·w9  
z3 = a11·w1 + a12·w2 + a13·w3 + a16·w4 + a17·w5 + a18·w6 + a21·w7 + a22·w8 + a23·w9  
z4 = a13·w1 + a14·w2 + a15·w3 + a18·w4 + a19·w5 + a20·w6 + a23·w7 + a24·w8 + a25·w9  
i.e. the forward pass

Now the backward pass -

This is the weight update rule - $w_i^* = w_i - \alpha \times \frac{\partial L}{\partial w_i}$  
wi is the old weight, $\alpha$ is learning rate and $\delta L/\delta wi$ is partial derivative of loss w.r.t. weight

now to calculate $\dfrac{\delta L}{\delta wi}$  we can use the chain rule as -  
Loss depends on $\hat{y}$ ,  $\hat{y}$ depends on the z (activation value), z depends on the weights of the kernel.  
So the change in weights *w* will result in change in values of *z*, change in values of *z* will result in change in values of $\hat{y}$, and that will further affect loss *L*.

$\dfrac{\delta L}{\delta w1}$ = $\dfrac{\delta L}{\delta \hat{y}}$ $\dfrac{\delta \hat{y}}{\delta z1}$ $\dfrac{\delta z1}{\delta w1}$ 

for simplification we will write this as -

$\dfrac{\delta L}{\delta w1}$ = $\dfrac{\delta L}{\delta z1}$ $\dfrac{\delta z1}{\delta w1}$

This is only for z1, as weight w1 also affects z2, z3, z4 we can write the equation as 

$\dfrac{\delta L}{\delta w1}$ = $\dfrac{\delta L}{\delta z1}$ $\dfrac{\delta z1}{\delta w1}$ + $\dfrac{\delta L}{\delta z2}$ $\dfrac{\delta z2}{\delta w1}$+ $\dfrac{\delta L}{\delta z3}$ $\dfrac{\delta z3}{\delta w1}$+ $\dfrac{\delta L}{\delta z4}$ $\dfrac{\delta z4}{\delta w1}$

The General formula is - 

$\dfrac{\delta L}{\delta wi}$ = $\dfrac{\delta L}{\delta zi}$ $\dfrac{\delta zi}{\delta wi}$ + $\dfrac{\delta L}{\delta z2}$ $\dfrac{\delta z2}{\delta wi}$+ $\dfrac{\delta L}{\delta z3}$ $\dfrac{\delta z3}{\delta wi}$+ $\dfrac{\delta L}{\delta z4}$ $\dfrac{\delta z4}{\delta wi}$

Simplifying the formula we can calculate the $\dfrac{\delta z1}{\delta w1}$

as z1 =a1·w1 + a2·w2 + a3·w3 + a6·w4 + a7·w5 + a8·w6 + a11·w7 + a12·w8 + a13·w9

therefore, $\dfrac{\delta z1}{\delta w1}$ = a1 and also $\dfrac{\delta z2}{\delta w1}$ = a3 , $\dfrac{\delta z3}{\delta w1}$ = a11 , $\dfrac{\delta z4}{\delta w1}$ = a13

so $\dfrac{\delta L}{\delta w1}$ and other partial derivatives become -

$\dfrac{\delta L}{\delta w1} = a1 . \dfrac{\delta L}{\delta z1} + a3 . \dfrac{\delta L}{\delta z2} + a11 . \dfrac{\delta L}{\delta z3} + a13 . \dfrac{\delta L}{\delta z4}$

$\dfrac{\delta L}{\delta w2} = a2 . \dfrac{\delta L}{\delta z1} + a4 . \dfrac{\delta L}{\delta z2} + a12 . \dfrac{\delta L}{\delta z3} + a14 . \dfrac{\delta L}{\delta z4}$

$\dfrac{\delta L}{\delta w3} = a3 . \dfrac{\delta L}{\delta z1} + a5 . \dfrac{\delta L}{\delta z2} + a13 . \dfrac{\delta L}{\delta z3} + a15 . \dfrac{\delta L}{\delta z4}$

$\dfrac{\delta L}{\delta w4} = a6 . \dfrac{\delta L}{\delta z1} + a8 . \dfrac{\delta L}{\delta z2} + a16 . \dfrac{\delta L}{\delta z3} + a18 . \dfrac{\delta L}{\delta z4}$

$\dfrac{\delta L}{\delta w5} = a7 . \dfrac{\delta L}{\delta z1} + a9 . \dfrac{\delta L}{\delta z2} + a17 . \dfrac{\delta L}{\delta z3} + a19 . \dfrac{\delta L}{\delta z4}$

$\dfrac{\delta L}{\delta w6} = a8 . \dfrac{\delta L}{\delta z1} + a10 . \dfrac{\delta L}{\delta z2} + a18 . \dfrac{\delta L}{\delta z3} + a20 . \dfrac{\delta L}{\delta z4}$

$\dfrac{\delta L}{\delta w7} = a11 . \dfrac{\delta L}{\delta z1} + a13 . \dfrac{\delta L}{\delta z2} + a21 . \dfrac{\delta L}{\delta z3} + a23 . \dfrac{\delta L}{\delta z4}$

$\dfrac{\delta L}{\delta w8} = a12 . \dfrac{\delta L}{\delta z1} + a14 . \dfrac{\delta L}{\delta z2} + a22 . \dfrac{\delta L}{\delta z3} + a24 . \dfrac{\delta L}{\delta z4}$

$\dfrac{\delta L}{\delta w9} = a13 . \dfrac{\delta L}{\delta z1} + a15 . \dfrac{\delta L}{\delta z2} + a23 . \dfrac{\delta L}{\delta z3} + a25 . \dfrac{\delta L}{\delta z4}$

![Pasted image 20260715134130.png](Attachments/Pasted%20image%2020260715134130.png)

i.e.  the weight update formula we seen earlier

![Pasted image 20260715134159.png](Attachments/Pasted%20image%2020260715134159.png)


Multiple filters - (consider 2 filters)  
After the convolution operation done by the 2 filters, we get 2 channels as result.  
We then perform maxpooling operation. the pool size is same as the filter size. We just get the max value from the window. Other pooling methods are min pooling and average pooling. The output is those 2 channels but reduced.

We flatten the value and pass it to fully connected layer.

Stacked filters -  
When filters are stacked, we simply multiply the values so that we get 2 matrices, then we do matrix multiplication, after adding the bias term we will get value of z and passing this value to activation function will give us the activation value a.


## Day 43 - Padding and Strides

Padding  
Why Padding is necessary -  
- When we perform a convolution operation, the data gets reduced and multiple operations will result in losing information.  
- The pixels in the corners will not be that much included as much as a pixel in the center, so the pixel in the corner will not affect the feature map that much. Padding helps to avoid loss of edge information

We add 2 rows (top and bottom) and 2 columns (right and left) usually with zeros. (Zero-Padding)  
If the image is n * n and the filter is f * f then with padding p=1, the shape of the feature map is (n+2p−f+1) * (n+2p−f+1)

Valid padding - no padding  
Same padding - we add padding such a way that output feature map has same dimensions as input.  
Here padding is defined as p = (f-1)/2 where f is size of filter

Strides -  
Stride (2,2) means we move 2 pixels in direction of x-axis and y-axis  
Strides reduces the size of feature map.

If the image is n * n and the filter is f * f then with no padding and stride = s, division here is floor operation

$$\dfrac{(n+2p-f)}{s} + 1,\dfrac{(n+2p-f)}{s} + 1$$  
Why Strides are necessary -  
- If we only want high level features  
- Higher strides, less computationally expensive training 


## Day 44 - Pooling Operation

Why Pooling is required -  
- Very large amount of memory is required  
- Translation Variance -  
	-  Image - filter - feature map -> relu(feature map) this makes it non linear and then we apply pooling (min, max, average, L2, global)  
	- size of pooling layer (2,2), stride (2), type of pooling (min, max)  
	- We are only taking dominant features from a receptive field.  
	- Eliminating the low level details and only taking high level details.

Pooling on volumes (multiple feature maps) will give us output with multiple channels

Advantages of Pooling -  
- Reduced size  
- Translation invariance - presence of feature matters instead of location of the feature  
- In case of max pooling, images get enhanced  
- No need of training, its just a aggregate operation.

Disadvantages of pooling -  
- Bad for image segmentation task  
- Loss of information

Types of pooling -  
- Max  
- Average  
- Global pooling -  
	1. Global max pooling (can be used instead of flatten, reduced overfitting)  
	2. Global average pooling
