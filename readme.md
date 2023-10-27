# CerP
In this repository, code is for our AAAI 2023 paper [Poisoning with Cerberus: Stealthy and Colluded Backdoor Attack against Federated Learning](https://ojs.aaai.org/index.php/AAAI/article/view/26083)

## Installation
Install Pytorch

### Reproduce experiments:

- we can use Visdom to monitor the training progress.
```
python -m visdom.server -p 8098
```

- run experiments for the CIFAR-100 dataset:
```
python main.py --params utils/X.yaml
```
`X` = `mkrum` or `bulyan`. 

Parameters can be changed in those yaml files to reproduce our experiments.



Stay tuned for further updates, thanks!

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{DBLP:conf/aaai/LyuHWLWL023,
  author       = {Xiaoting Lyu and
                  Yufei Han and
                  Wei Wang and
                  Jingkai Liu and
                  Bin Wang and
                  Jiqiang Liu and
                  Xiangliang Zhang},
  title        = {Poisoning with Cerberus: Stealthy and Colluded Backdoor Attack against
                  Federated Learning},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                  7-14, 2023},
  pages        = {9020--9028},
  publisher    = {{AAAI} Press},
  year         = {2023},
  url          = {https://doi.org/10.1609/aaai.v37i7.26083},
  doi          = {10.1609/AAAI.V37I7.26083},
  timestamp    = {Mon, 04 Sep 2023 16:50:26 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/LyuHWLWL023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
## Acknowledgement 
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
- [AI-secure/DBA](https://github.com/AI-secure/DBA)