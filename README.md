# Learning effective pruning at initialization from iterative pruning

Liu, Shengkai* and Cheng, Yaofeng* and Zha, Fusheng and Guo, Wei and Sun, Lining and Bing, Zhenshan and Yang, Chenguang
(*: equal contribution)
  
Pruning at initialization (PaI) reduces training costs by removing weights before training, which becomes increasingly crucial with the growing network size. However, current PaI methods still have a large accuracy gap with iterative pruning, especially at high sparsity levels. This raises an intriguing question: can we get inspiration from iterative pruning to improve the PaI performance? In the lottery ticket hypothesis, the iterative rewind pruning (IRP) finds subnetworks retroactively by rewinding the parameter to the original initialization in every pruning iteration, which means all the subnetworks are based on the initial state. Here, we hypothesise the surviving subnetworks are more important and bridge the initial feature and their surviving score as the PaI criterion. We employ an end-to-end neural network (**AutoS**parse) to learn this correlation, input the model's initial features, output their score and then prune the lowest score parameters before training. To validate the accuracy and generalization of our method, we performed PaI across various models. Results show that our approach outperforms existing methods in high-sparsity settings. Notably, as the underlying logic of model pruning is consistent in different models, only one-time IRP on one model is needed (e.g., once IRP on ResNet-18/CIFAR-10, AutoS can be generalized to VGG-16/CIFAR-10, ResNet-18/TinyImageNet, et al.). As the first neural network-based PaI method, we conduct extensive experiments to validate the factors influencing this approach. These results reveal the learning tendencies of neural networks and provide new insights into our understanding and research of PaI from a practical perspective.

# Citation
```
@article{liu2024learning,
  title={Learning effective pruning at initialization from iterative pruning},
  author={Liu, Shengkai and Cheng, Yaofeng and Zha, Fusheng and Guo, Wei and Sun, Lining and Bing, Zhenshan and Yang, Chenguang},
  journal={arXiv preprint arXiv:2408.14757},
  year={2024}
}
```

