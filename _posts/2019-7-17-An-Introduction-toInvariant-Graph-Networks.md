This is the first post summarizing the main ideas and constructions in a series of three recent papers (Maron et al., 2019 a,b,c) introducing and investigating a novel type of neural networks for learning irregular data such as graphs and hypergraphs. In this post we focus on (Maron et al., 2019a) that was presented at ICLR 2019  

## Algebraic view of convolutional neural networks.
The goal of this note is presenting a family of neural network architectures suitable for learning irregular data in the form of graphs, or more generally, hypergraphs. This family presents a tradeoff between expressivity (i.e., the ability to approximate a large and complicated set of functions), and efficiency (i.e., the amount of time and space resources used by these architectures).

![Image](https://haggaim.github.io/images/2019-7-17/image001.png)

Image credit: hypergraph - Wikipedia d

The main idea (see right image)  is to adapt the concept of image convolutions, as a means of
dramatically reducing the number of parameters in a neural network, to graph and hypergraph data. In more detail, translations of images are transformations  that do not change the image
content, see e.g., the image above. Hence, most functions  one is interested to learn on images, like image classification, will be invariant to translations, namely will satisfy  for all translations , where  represents the image, and  the application of the translation  to the image.  

![Image](https://haggaim.github.io/images/2019-7-17/image005.png)

Image credit: imgur - https://imgur.com/mEIUqT8

A Multi-Layer perceptron (MLP) is a general-purpose architecture that can approximate any continuous function (Cybenko 1989, Hornik 1991). The architecture is composed of multiple layers , where each layer has the form , where  is the input to the layers,   is a non-linear function applied entry-wise (e.g., ReLU), and  is a linear (in fact, affine) function.  are the tuneable parameters of the network. Using an MLP to learn functions on images is daunting: consider a low-resolution image of 100x100x3, and let the output of the first layer be of the same dimension 100x100x3 (say we want to apply some transformations to the colors of the image). This would make the parameter  of dimension 10^9 (approximately) and would only represent the linear part of a single layer.  

![Image](https://haggaim.github.io/images/2019-7-17/image017.png)


Motivated by the fact that we are looking to approximate invariant functions, a reasonable way to try and reduce this huge number of parameters, is to replace the general linear (in fact, affine) transformations  with linear transformations that are themselves invariant to translations, namely, satisfy  for all . Since translations are transitive, i.e., can map any pixel to any other pixel,  belongs to a one-dimensional vector space proportional to the sum operator. In other words, in this case, all the network can do is summing all the pixel values (for each feature dimension). Clearly, using only the sum operator would never lead to useful neural network models as it will not distinguish between an image and its arbitrarily scrambled version (i.e., arbitrarily ordered pixels).

![Image](https://haggaim.github.io/images/2019-7-17/image030.png)

A much more useful idea is to think about equivariant linear operators, namely linear operators that commute with the translations (see image on the left), mathematically satisfying   for all .  This condition implies that is a convolution operator (in fact, equivariance is a defining property of convolutions) and  is a constant vector. 

![Image](https://haggaim.github.io/images/2019-7-17/image040.png)

Now, a neural network defined by composing several equivariant layers , followed by a single invariant layer, , and an MLP, , namely  (see image), is by construction an invariant function and offers a much richer and expressive model that was used successfully to learn complicated image functions via Convolutional Neural Networks (CNNs) (Krizhevsky et al, 2012) that roughly follow this construction (the main difference is that CNNs usually use spatial pooling layers, that are not necessarily relevant for this note).

![Image](https://haggaim.github.io/images/2019-7-17/image047.png)

## Representing graphs as tensors
Instead of images, we would like to learn graphs, or more generally hypergraphs. Graphs and hypergraphs are mathematical objects that are widely used for representing structures ranging from social networks on the one hand to molecules, on the other hand. A graph can be defined as a set of  elements (nodes) for which we have some information  attached to its i-th element, and some information attached to pairs of elements (edge),  will denote the information attached to the pair consisting of the i-th and j-th nodes. We will encode this data using a tensor , where the diagonal elements  encode the node data and the off-diagonal elements, , , the edge data (for clarity, we discuss a single feature dimension). A natural generalization of a graph is a hypergraph where information is attached not only to single elements and pairs of elements but also 3-tuples, 4-tuples, or in general k-tuples. We represent such data using , and each entry  represents the information of the corresponding k-tuple of elements.  The images depict a simple graph and its tensor representation (matrix, top row) and a hypergraph of order 3 and its representing tensor (bottom row).

![Image](https://haggaim.github.io/images/2019-7-17/image064.png)


## Symmetries of graphs
Transformations that do “not change” the input data  will be called symmetries.  Translations are symmetries of images, but different kind of data, such as graphs may exhibit other symmetries. Note that in the representations of graphs introduced above, one can choose a different ordering of the set of nodes which affect the resulting tensor representation . 

![Image](https://haggaim.github.io/images/2019-7-17/image074.png)


Two graphs  will be considered as the same (a.k.a. isomorphic) if there exists a permutation  so that , where  is a rearrangement of the rows and columns of  according to , that is, .  Using  and not  in this definition is to make this action a left action, but is pretty arbitrary and does not really matter in our discussion. See the inset image, where  is the permutation matrix representing the permutation . These symmetries generalize to hypergraphs where the permutation  applied to all dimensions of , namely . This action is visualised for an example of a 3-order tensor in the left image. Therefore, the symmetries of graph data are represented via the permutation group.

![Image](https://haggaim.github.io/images/2019-7-17/image104.png)


## Invariant graph networks
We will use our understanding of the graph symmetries to come up with an effective inductive bias to learn graph data. That is, a way to restrict the linear transformations in the MLP so to achieve neural networks that are by construction invariant to the graph symmetries, without compromising the expressive power of the model. Similarly to the image case discussed above, trying to consider linear transformations of graph data  that are invariant, i.e., , leads to a poor space of operators: basically such  are operators belong to a two-dimensional vector space containing summing the diagonal and summing the off diagonal of . So for instance, networks using only such invariant layers could not distinguish two graphs if they have the same number of nodes and edges, which is of course not satisfying for learning interesting functions of graphs.  
Exactly as in the image case, remedy comes from considering (the larger space of) equivariant operators, namely linear operators satisfying  for all . As for images, we will consider neural networks defined by composing several equivariant layers , followed by a single invariant layer, , and an MLP, , namely .  is by construction an invariant function, namely satisfies . We call this network **Invariant Graph Network** (IGN) and discuss its properties next, but first we need to characterize the space of equivariant and invariant linear maps between tensors.

![Image](https://haggaim.github.io/images/2019-7-17/image116.png)


Note that in an IGN the hidden variables can be arbitrary tensors, , even when the input tensors are of a lower order than . Indeed, equivariant linear operators can map between different order tensors . For example, consider an IGN that receives an input graph tensor but learns information attached to triplets of nodes, then it would require a layer mapping order-2 tensor (i.e., matrix), ,  to order-3 tensor, . Another example is a network that takes in only a set information (data only on nodes) in  and learns information on pairs of elements, . From now on, the term -IGN will be used for describing an IGN with a maximal inner tensor degree of .
## Linear equivariant/invariant operators and the fixed point equations
We are looking to characterize affine transformations  equivariant () or invariant () to the permutation action , as defined above. This is done in (Maron et al., 2019a) where the key idea is the following. Ignoring the constant part (i.e., bias) of  (that can be treated in a similar way), a linear transformation  can be encoded as a tensor . This is similar to a linear transformation  represented using a matrix in . After applying some algebraic manipulations, the equations  can be expressed compactly as ; this includes both the case of linear equivariant operators (), and linear invariant operators (). We name these equations the fixed point equations as the space of equivariant/invariant operators  is represented by all tensors  that are fixed by the action of the permutation group . 
How do we solve the fixed point equations? Any solution  should be constant along orbits of the permutation group. For example, let us consider , which corresponds to equivariant operators . If we consider , that is, the permutation that interchanges 1 and 2 and keeps all other values in  at place. Then, .  Similarly, considering  we get . Continuing in this manner we get that the diagonal  is constant. Using a similar argument, one can see that , for example, would equal all entries of the form , where  are all different. In general  will be constant along indices  that have the same equality pattern, that is, indices that preserve the equality and inequality relations between pairs. Therefore, the number of different orbits in this case will be the number of different equality patterns of four indices which equals the number of partitions of a set with four elements, also known as the Bell Number; in this case, . See image below.

![Image](https://haggaim.github.io/images/2019-7-17/image197.png)

An orthogonal basis to the equivariant operators can, in turn, be constructed by considering an equality pattern , e.g., , and the tensor  . As shown in (Maron et al., 2019a) applying the equivariant operators can be done efficiently in  operations. As a result, a linear equivariant operator has the form , where  are the learnable parameters of the model. The figure illustrates the 15 different basis elements for . Here, each square represents a matrix operating on the column stack of an  input tensor. Black pixels represent zero values, while white pixels represent the value 1. 
![Image](https://haggaim.github.io/images/2019-7-17/image197.png)


In the general case, solving the fixed point equations for  reduces to a fixed point equation for  . This equation can be solved by using the method above. In this case, we will have  basis elements (the number of equality patterns on  indices) and the basis is given by the indicator tensors  of these equality patterns.

![Image](https://haggaim.github.io/images/2019-7-17/image226.png)


**Remarks.** (Maron et al., 2019a) can be seen as a generalization of Deep Sets (Zaheer et al., 2017, Qi et al,. 2017) that dealt with the case of node features and (Hartford et al., 2018) that studied equivariant layers for interaction between multiple sets. (Kondor et al., 2018) also identified several linear and quadratic equivariant operators and showed that the resulting network can achieve excellent results on popular graph learning benchmarks.
## How expressive are IGNs?
Up until now, we have discussed a principled way of constructing invariant graph networks. Our next step is analyzing the expressive power of these models. Expressivity of graph neural networks is studied from different perspectives (Chen et al., 2019). In (Maron et al., 2019a,b) we studied the question: what kind of functions can IGNs approximate? More recently, in (Maron et al., 2019c), we study the capability of IGNs to discriminate non-isomorphic graphs. When we say that a model can discriminate between two non-isomorphic graphs  we mean that there is a set of parameters for the IGN model that results in a function  with the property . Remember that by construction  if  are isomorphic.
**Function approximation point of view.** In (Maron et al., 2019a) we proved the first result of this kind:  2-IGNs, i.e., an IGN with a maximal tensor degree of 2, can approximate any message passing neural network to an arbitrary precision (on compact sets). Message passing neural networks (Gilmer et al., 2017) are very popular models for learning graph data which currently provide state of the art results on various graph learning benchmarks.
In (Maron et al., 2019b) we further generalise this result and show that these networks are universal when using sufficiently large tensor order . Universality means that -IGNs can approximate any continuous invariant function to an arbitrary precision (on compact sets). The main idea is showing that IGNs can approximate a set of generating invariant polynomials (see theorems (1)-(2) in (Kraft and Procesi, 2000)). An alternative proof was recently suggested in (Keriven and Peyré, 2019).
**Graph discrimination point of view.** In (Maron et al., 2019c) we show that -IGNs are tightly related to a hierarchy of graph isomorphism tests called the Weisfeiler-Leman (WL) hierarchy. The WL hierarchy defines an algorithm, called k-WL, for every . The first member of this hierarchy, 1-WL (also called the color refinement algorithm) iteratively generates node features by aggregating features of neighboring nodes. When this process reaches a steady state (i.e., same color nodes are not colored by different new colors), the histogram of these colors serves as a global graph descriptor that can be compared to descriptors of other graphs in order to test isomorphism. For , -WL is defined in a similar way where instead of node features node -tuples features are used. It can be shown that for any , -WL is strictly more powerful than -WL, meaning it can discriminate a broader set of non-isomorphic graphs.

This direction follows the prominent works of (Morris et al., 2019; Xu et al., 2019) that prove that message passing neural networks can discriminate between non-isomorphic graphs as well as the 1-WL. Sadly, 1-WL is not strong enough and, for example, cannot discriminate regular graphs such as the pair of graphs in the figure. [Morris et al., 2019] also suggests a message passing algorithm directly on -tuples to achieve -WL discrimination power. 

![Image](https://haggaim.github.io/images/2019-7-17/image226.png)

In (Maron et al., 2019) we prove that -IGNs can discriminate graphs at least as good as the -WL algorithm, for every . For example, 3-WL already discriminates the pair regular graphs mentioned above.
**A simple and expressive variant of IGNs.** The main drawback of -IGNs is the fact that one needs to store and process -order tensors, limiting its applicability. In (Maron et al., 2019c) we suggest a variant of 2-IGNs that is at least as expressive as the 3-WL test. Hence, strictly more expressive than message passing models. This variant, which we call 2-IGN+, consists of building blocks as follows (see inset): Each block is constructed with 3 different MLPs applied to the feature dimension of the input 2-tensor. The outputs of two of these MLPs are multiplied feature-wise using regular matrix multiplication, while the output of the third MLP is concatenated to this product output. One of the main strengths of this model is its simplicity: MLPs on the feature dimension can be implemented by simple image convolution, while matrix multiplication is a basic operation that is supported by all deep learning frameworks. 

![Image](https://haggaim.github.io/images/2019-7-17/image252.png)

Note that applying an MLP to the feature dimension can be seen as a particular simplified instance of IGN, using only a single linear equivariant operator, namely, scaled identity. When the feature dimension is one, scaled identity is a linear operator that takes as input  and outputs , . When the feature dimension of  is greater than one,  becomes a matrix that is applied to the feature dimension. 
Let us provide some intuition as to why matrix multiplication improves expressiveness. We will do that by showing matrix multiplication allows this model to distinguish between the two regular graphs discussed above, which are 1-WL (and 2-WL) indistinguishable. Consider the case the input tensor  representing a graph  holds the adjacency matrix. We can build a network with 2 blocks computing  and then take the trace of this matrix (using the invariant layer). Recall that the -th power of the adjacency matrix computes the number of -paths between vertices; in particular,  computes the number of cycles of length 3. Counting shows the right graph has 0 such cycles while the left graph has 12. 
**Summary of the IGN expressiveness.** The following figure illustrates the expressiveness results for IGNs. It provides an overview of the main tradeoffs between efficiency and approximation power.
![Image](https://haggaim.github.io/images/2019-7-17/image252.png)

Written by Haggai Maron and Yaron Lipman, *Weizmann Institute of Science*

## Bibliography

(Cybenko, 1989) Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303–314. 

(Hartford et al., 2018) Hartford, J. S., Graham, D. R., Leyton-Brown, K., and Ravanbakhsh, S. (2018). Deep models of interactions across sets. In ICML. 

(Hornik, 1991) Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. Neural networks, 4(2):251–257. 

(Kondor et al., 2018) Kondor, R., Son, H. T., Pan, H., Anderson, B., and Trivedi, S. (2018). Covariant compositional networks for learning graphs. arXiv preprint arXiv:1801.02144. 

(Krizhevsky et al., 2012) Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105. 

(Maron et al., 2019a) Maron, H., Ben-Hamu, H., Shamir, N., and Lipman, Y. (2019a). Invariant and equivariant graph networks. In International Conference on Learning Representations. 

(Maron et al., 2019b) Maron, H., Fetaya, E., Segol, N., and Lipman, Y. (2019b). On the universality of invariant networks. In International conference on machine learning. 

(Maron et al., 2019c) Maron, H., Ben-Hamu, H., Serviansky, H., and Lipman, Y. (2019c). Provably powerful graph networks. arXiv preprint arXiv:1905.11136.

(Qi et al., 2017) Qi, C. R., Su, H., Mo, K., and Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. In Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017. 

(Zaheer et al., 2017) Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., and Smola, A. J. (2017). Deep sets. In Advances in Neural Information Processing Systems, pages 3391–3401


