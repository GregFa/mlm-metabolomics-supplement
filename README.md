# Matrix Linear Models for connecting metabolite composition to individual characteristics


Gregory Farage<sup>1</sup>, Chenhao Zhao<sup>1</sup>, Hyo Young Choi<sup>1</sup>, Timothy J. Garrett<sup>2</sup>, Katerina Kechris<sup>3</sup>, Marshall B. Elam<sup>4</sup>, Śaunak Sen<sup>1</sup>

><sup>1</sup>Department of Preventive Medicine, College of Medicine, University of Tennessee Health Science Center, Memphis, TN   
<sup>2</sup>Department of Pathology, Immunology and Laboratory Medicine, University of Florida, Gainesville, FL    
<sup>3</sup>Department of Biostatistics & Informatics, Colorado School of Public Health, University of Colorado Anschutz Medical Campus, Aurora, CO   
<sup>4</sup>Department of Pharmacology and of Medicine, University of Tennessee Health Science Center, Memphis, TN   


### Abstract     
High-throughput metabolomics data provide a detailed molecular window into biological processes. We consider the problem of assessing how the association of metabolite levels with individual (sample) characteristics such as sex or treatment may depend on metabolite characteristics such as pathway. Typically this is one in a two-step process: In the first step we assess the association of each metabolite with individual characteristics. In the second step an enrichment analysis is performed by metabolite characteristics among significant associations. We combine the two steps using a bilinear model based on the matrix linear model (MLM) framework we have previously developed for high-throughput genetic screens. Our framework can estimate relationships in metabolites sharing known characteristics, whether categorical (such as type of lipid or pathway) or numerical (such as number of double bonds in triglycerides). We demonstrate how MLM offers flexibility and interpretability by applying our method to three metabolomic studies. We show that our approach can separate the contribution of the overlapping triglycerides characteristics, such as the number of double bonds and the number of carbon atoms. The proposed method have been implemented in the open-source Julia package, MatrixLM. Data analysis scripts with example data analyses are also available.

### Materials

- [Pluto notebook Analysis]()

### References:

- Liang, J. W., Nichols, R. J., & Sen, Ś. (2019). Matrix linear models for high-throughput chemical genetic screens. Genetics, 212(4), 1063–1073. doi: [10.1534/genetics.119.302299.](https://academic.oup.com/genetics/article/212/4/1063/5931246)
- Liang, J. W. & Sen, Ś. (2021). Sparse matrix linear models for structured high-throughput data. To appear in The Annals of Applied Statistics. [arXiv preprint: arXiv:1712.05767 [stat.CO].](https://arxiv.org/abs/1712.05767)


### Resources:

- [MatrixLM.jl package](https://github.com/senresearch/MatrixLM.jl)
- [MatrixLMnet.jl](https://github.com/senresearch/MatrixLMnet.jl)
- [Sen Research Group Resources](https://senresearch.github.io/)