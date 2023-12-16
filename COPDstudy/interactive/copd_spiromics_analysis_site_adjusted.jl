### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 148df6fb-feac-4fe2-9ccc-18cb460796bf
begin
	#############
	# Libraries #
	#############
	using CSV, DataFrames, DataFramesMeta, Missings#, CategoricalArrays
	using StatsBase, Statistics, MatrixLM
	using Random, Distributions, StatsModels#, MultivariateStats 
	using LinearAlgebra, PrettyTables, Latexify
	using FreqTables, Plots, StatsPlots, PlutoUI, Images, FileIO
	using Plots.PlotMeasures, ColorSchemes, RecipesBase#, PlotlyBase
	using MultipleTesting


	######################
	# External functions #
	######################
	include(joinpath(@__DIR__, "..","src","wrangle_utils.jl" ));
	include(joinpath(@__DIR__, "..","src","utils.jl" ));
	include(joinpath(@__DIR__, "..","src","utils_copd_spiro.jl" ));
	include(joinpath(@__DIR__, "..","src","demog.jl" ));
	include(joinpath(@__DIR__, "..","src","mLinearModel.jl" ));
	include(joinpath(@__DIR__,"..", "src","myPlots.jl" ));
	
end;

# ╔═╡ d9a9dd61-4cab-4ee0-a769-7dfc45c281b5
PlutoUI.TableOfContents()

# ╔═╡ 34d05aeb-63fc-46c5-aefe-b9f6622e6c9b
md"""
# COPD-SPIROMICS cohort Analysis Using Matrix Linear Models  
---
**Gregory Farage, Śaunak Sen**    

    gfarage@uthsc.edu / sen@uthsc.edu
    Division of Biostatistics
    Department of Preventive Medicine
    University of Tennessee Health Science Center
    Memphis, TN
"""

# ╔═╡ a59b3bc2-edaa-11ec-3b2d-fd5bdd0ee1f2
md"""
## Background

Susceptibility and progression of chronic obstructive pulmonary disease (COPD) and response to treatment often differ by sex. Yet, the metabolic mechanisms driving these sex-specific differences are still poorly understood.

Accurately characterizing the sex-specific molecular differences in COPD is vital for personalized diagnostics and therapeutics.

Metabolic signatures of sex differences in COPD may show variation in the pathways.

## Project Summary

This project seeks to investigate the association between metabolite characteristics (e.g., pathways) and patient characteristics such as sex or age by using Matrix Linear Models.

## Methods
 $(load("../images/matrixlinearmodel.png"))    
     
Matrix linear models provide a framework for studying associations in high-throughput data using bilinear models.

$$Y= X B Z^\prime + E,$$

where
- ``Y`` is the outcome data matrix (metabolites levels)
- ``X`` is a design matrix constructed from individual covariates (phenotype status, sex, age, BMI...)
- ``Z`` is a design matrix constructed from outcome covariates (information about the pathways)
- ``E`` is a random error assumed to have a mean zero, independent across individuals, but possibly correlated across outcomes

$$V(vec(E)) = \Sigma \otimes I.$$

The model can also be viewed as a bilinear model of $y$ in terms of the $x$ and $z$ covariates:

$$y_{ij} = \sum_{i{=}1}^n \sum_{j{=}1}^m b_{kl} x_{ik} z_{jl} + e_{ij}.$$

The regression coefficients ($b$'s) can be interpreted as interactions between the corresponding $x$ and $z$ covariates.

"""

# ╔═╡ 58cdac80-d63b-4c3e-9193-d507a0da9be6
md"""
## Data Analysis
"""

# ╔═╡ de5b8917-915c-4c17-8d2c-d55f06822a3f
begin
	# Get cohort data
	copd =  get_data("COPDGene")
	spiro = get_data("SPIROMICS")

	
	#######################
	# Plotting attributes #
	#######################
	myfont = "Helvetica"
	mytitlefontsize = 12 
	
end;

# ╔═╡ 9da0827e-aec0-4981-9139-aee445104a04
md"""
### Individual Characteristics
"""

# ╔═╡ 4e368004-d055-40a1-a28d-c0f4ca55dad7
md"""
The following table contains  the clinical data dictionary:
"""

# ╔═╡ a5ce8064-1af1-47cc-8dcd-db7d96f3d537
begin
	fileClinicalDict = joinpath(@__DIR__,"..","data","processed","COPDGene",
								"ClinicalDataDictionary.csv");
	dfClinicalDict = CSV.read(fileClinicalDict, DataFrame);
	latexify(dfClinicalDict; env=:mdtable, latex=false)
end

# ╔═╡ e176af8a-8bc3-497d-91f8-aa7259ab3c48
md"""
\* *Notes:*
*Gold staging system describes how severe COPD is: 1-early, 2-moderate, 3-severe, 4-very severe; forced expiratory volume (FEV1) shows how much air can be exhaled from lungs in 1 second; forced vital capacity (FVC) is the largest amount of air that can be breathed out after breathing in as deeply as possible; FEV1/FVC ratio is the Tiffeneau-Pinelli index.*     
"""

# ╔═╡ 32b95e81-6761-4b2b-a59b-1eedc9a1443e
md"""
##### COPDGene
"""

# ╔═╡ ae4fd58c-cd3d-4b7d-8c29-dd84e2a103c9
md"""
The following table presents the demographics for the COPDGene cohort that contains *n*= $(size(copd.dfInd, 1)) participants:
"""

# ╔═╡ 96741af6-faa9-40ef-85e2-8647199ed6ab
get_demog_table(copd.dfInd)	

# ╔═╡ 1f367dfd-8642-4c2c-806b-42388725f0cf
md"""
\* *Notes:*
*continuous variables show mean(standard deviation); categorical variables show total(%).*     
"""

# ╔═╡ 19b7b087-ce29-43f4-b870-299233045d3a
md"""
##### SPIROMICS
"""

# ╔═╡ 3f703537-ef4f-4d2d-a348-6ac4332abdaa
md"""
The following table presents the demographics for the SPIROMICS cohort that contains *n*= $(size(spiro.dfInd, 1)) participants:
"""

# ╔═╡ 265e748a-f17d-4a52-a6b7-4c0537fccc86
get_demog_table(spiro.dfInd)	

# ╔═╡ 8ef72518-4b30-4efe-9415-d84719d39be4
md"""
\* *Notes:*
*continuous variables show mean(standard deviation); categorical variables show total(%).*     
"""

# ╔═╡ f679bf30-78a6-48c0-a17d-391a0e185289
md"""
### Metabolites Characteristics
"""

# ╔═╡ 3e5e2657-253c-4f96-a5f3-3aea07485478
md"""
#### Preprocessing
We preprocessed the metabolomic datasets through the following steps: (1) we applied a KNN imputation (Hastie *et al.*, 1999) using the nearest neighbor averaging method from the R package `impute` (version 1.46.0), (2) we performed a normalization based on the probabilistic quotient normalization (Dieterle *et al.*, 2006) method, (3) we applied a log2-transformation to make data more symmetric and homoscedastic.
"""

# ╔═╡ 01f9a0b1-4d33-49a5-bf7a-fec0b04ad1a6
md"""
#### Dataset Overview 
"""

# ╔═╡ 6902142b-a049-4724-aa1b-ee5b167c0cd6
md"""
##### COPDGene 
The metabolomic dataset contains *m*=$(size(copd.dfMet, 2)-1) metabolites. According to their attributes, each metabolite may belong to one of the $(length(unique(copd.dfRef.SuperPathway))-1) identified classes or to the untargeted class, named "NA". It is also possible to sub-classified them based on  $(length(unique(copd.dfRef.SubPathway))-1) sub-classes and one untargeted sub-class, tagged as "NA". The following table presents the $(length(unique(copd.dfRef.SuperPathway))) "super classes" and their ID tags: 
"""

# ╔═╡ 859546bf-7508-45ab-b10e-412314b6d753
get_superclass_table(copd.dfRef)

# ╔═╡ 521b36b1-965a-483e-b67c-b78af8ea129b
md"""
The following table presents the $(length(unique(copd.dfRef.SubPathway))) sub-classes and their ID tags: 
"""

# ╔═╡ 5e0c72da-44db-4f92-9002-03682ac3843c
get_subclass_table(copd.dfRef)

# ╔═╡ ade4dd1e-99c8-4948-9922-21241c29561f
md"""
##### SPIROMICS 
The metabolomic dataset contains *m*=$(size(spiro.dfMet, 2)-1) metabolites. According to their attributes, each metabolite may belong to one of the $(length(unique(spiro.dfRef.SuperPathway))-1) identified classes or to the untargeted class, named "NA". It is also possible to sub-classified them based on  $(length(unique(spiro.dfRef.SubPathway))-1) sub-classes and one untargeted sub-class, tagged as "NA". The following table presents the $(length(unique(spiro.dfRef.SuperPathway))) "super classes" and their ID tags: 
"""

# ╔═╡ 764dd801-f96c-4d9b-95df-27f6e339bb41
get_superclass_table(spiro.dfRef)

# ╔═╡ 4c0a47e9-8dc5-4635-9449-39dfd3b1eee5
md"""
The following table presents the $(length(unique(spiro.dfRef.SubPathway))) sub-classes and their ID tags: 
"""

# ╔═╡ c5b66499-9a39-4181-b177-6c3b0c563fc9
get_subclass_table(spiro.dfRef)

# ╔═╡ 6952ec52-982d-4a37-ae1a-e7f5f1972fda
md"""
## Modeling Decision

$$Y= X B Z^\prime + E,$$
"""

# ╔═╡ 1f9d797d-5832-4e28-bfe5-056cf3e70a33
md"""
### X matrix
"""

# ╔═╡ 85236909-e107-4aa7-8b19-57892f4610d0
@bind xCovariates MultiCheckBox(vcat(["Intercept"], names(copd.dfInd)[5:end]), 
								default = ["Intercept", "Sex", "Age", "BMI",
								"SmokingPackYears", "FEV1pp", "PercentEmphysema",
								"COPD", "NHW", "CurrentSmoker"])



# ╔═╡ 5981d0b3-3de7-43cf-af1f-51cfc0179014
# @bind xCovariates2 MultiCheckBox(vcat(["Intercept"], names(spiro.dfInd)[3:end]), 
# 								default = ["Intercept", "Sex"])

# ╔═╡ 807aa3cf-0898-4e20-8b19-d5627d4050e1
if (["Sex"] ⊆ xCovariates) && (length(xCovariates) > 1 && xCovariates != ["Intercept", "Sex"])
# md"Include interactions with `Sex` $(@bind hasInteractions CheckBox(default=true))"
	@bind hasInteractions Radio(["Sex Interactions", "No Sex Interactions"], 
								default = "No Sex Interactions")	
end

# ╔═╡ 1d314d3f-3b2a-4e6f-9193-a1e0bd456a93
begin 
	vPredictorNames = copy(xCovariates)
	

	if (@isdefined hasInteractions) && (hasInteractions == "Sex Interactions" )
		idxNotSex = findall(xCovariates.!= "Sex" .&& xCovariates.!= "Intercept")
		vPredictorNames[idxNotSex] .= "Sex*".*xCovariates[idxNotSex]
		vPredictorNames = vPredictorNames[Not(findall(vPredictorNames.=="Sex"))]
	else
		vPredictorNames = copy(xCovariates)
		
	end

	if ["Intercept"] ⊆ xCovariates
		vPredictorNames[findall(vPredictorNames .=="Intercept")] .= "1"
	end
	
	frml = join(vPredictorNames, " + ")
	
md"""
Replace `+` by `*` for interaction:

$(@bind tf_frml TextField(default=frml))

"""	

end


# ╔═╡ 5a5ed093-fa57-4666-bcf5-3e30f1a1182f
@bind bttnModel Button("Confirm")

# ╔═╡ 69389cae-cc6e-4be4-8f54-3d7d0f68c0ed
frml_c = Ref("");

# ╔═╡ 9bfc2ec5-a235-43c6-a0af-d7287dfef304
begin
bttnModel	
frml_c[] = @eval $(Meta.parse("tf_frml"))
	# tf_frml 
md"""
The @formula design is: $(frml_c[])    
"""		
end

# ╔═╡ 27cee0f5-b86f-404a-9dd8-61bdf72bc43f
begin 
	bttnModel
	contrasts_copd = Dict(
		:Sex => EffectsCoding(base = sort(unique(copd.dfInd.Sex))[2]), 
		:NHW => EffectsCoding(base = sort(unique(copd.dfInd.NHW))[1]),
		:Site => EffectsCoding(base = sort(unique(copd.dfInd.Site))[1]),
		:CurrentSmoker => EffectsCoding(base = sort(unique(copd.dfInd.CurrentSmoker))[1]),
		# :FinalGold => EffectsCoding(base = sort(unique(dfInd.FinalGold))[1]),
		:COPD => EffectsCoding(base = sort(unique(copd.dfInd.COPD))[1])
	)
	contrasts_spiro = Dict(
		:Sex => EffectsCoding(base = sort(unique(copd.dfInd.Sex))[2]), 
		:NHW => EffectsCoding(base = sort(unique(copd.dfInd.NHW))[1]),
		:Site => EffectsCoding(base = sort(unique(spiro.dfInd.Site))[1]),
		:CurrentSmoker => EffectsCoding(base = sort(unique(copd.dfInd.CurrentSmoker))[1]),
		# :FinalGold => EffectsCoding(base = sort(unique(dfInd.FinalGold))[1]),
		:COPD => EffectsCoding(base = sort(unique(copd.dfInd.COPD))[1])
	)
	
	if frml_c[] != ""
		frml_site_adjusted = frml_c[]*" + Site"
		formulaX = eval(Meta.parse(string("@formula(0 ~ ", frml_site_adjusted, ").rhs")))
		if occursin("+", frml_site_adjusted) || occursin("*", frml_site_adjusted)
			vCovarNames = collect(string.(formulaX))
		else
			vCovarNames = [string(formulaX)]	
		end
		
		if ["1"] ⊆ vCovarNames
			vCovarNames[findall(vCovarNames .== "1")] .= "Intercept"
		end
		
		idx2change = findall(occursin.("&",vCovarNames))
		vCovarNames[idx2change]  .= replace.(vCovarNames[idx2change], " & " => "Ξ")

		mXcopd = modelmatrix(formulaX, copd.dfInd, hints = contrasts_copd)
	    mXspiro = modelmatrix(formulaX, spiro.dfInd, hints = contrasts_spiro) 
		
		vFrmlNames = Vector{String}()
	
	 	if frml_c[] == "1"
			# vFrmlNames = ["(Intercept)"]
			vPseudoFrmlNames_copd = ["(Intercept)"]
			vPseudoFrmlNames_spiro = ["(Intercept)"]
		else
			sch_copd = schema(formulaX, copd.dfInd, contrasts_copd)
			vFrmlNames_copd = apply_schema(formulaX, sch_copd) |> coefnames

			sch_spiro = schema(formulaX, spiro.dfInd, contrasts_spiro)
			vFrmlNames_spiro = apply_schema(formulaX, sch_spiro) |> coefnames
			# vFrmlNames = ModelFrame(eval(Meta.parse(string("@formula(0 ~ ", frml_c[], ")"))), dfInd, contrasts = contrasts) |> coefnames
		end
		
		function fix_covar_name(s::String)
			s = replace(s, "("=>"", ")"=>"", ": "=>"_")
			s = replace(s, " & " => "Ξ")
		end
		
		vPseudoFrmlNames_copd = fix_covar_name.(vFrmlNames_copd)
		vPseudoFrmlNames_spiro = fix_covar_name.(vFrmlNames_spiro)

		idx_covar_copd = findall(.!occursin.("Site", vFrmlNames_copd));
		idx_covar_spiro = findall(.!occursin.("Site", vFrmlNames_spiro));
			
		show(vFrmlNames_copd[idx_covar_copd])
		
	end
end

# ╔═╡ b3278187-f10b-4f16-a8b3-f1bd4e0ad854
begin	
if (@isdefined mXcopd)

	md"""
	Show COPDGene X matrix $(@bind radio_Xcopd CheckBox())
	"""		
end	
end

# ╔═╡ 49d9f10e-e9a2-4541-9418-e67040592dad
begin 
if (@isdefined mXcopd)
	if radio_Xcopd
		mXcopd
	else
		md"The size of COPDGene X is $(size(mXcopd, 1)) x $(size(mXcopd, 2))."
	end
end
end

# ╔═╡ 6b9213b8-d7ac-44c1-b0ae-a5e19e8e3d45
begin	
if (@isdefined mXspiro)

	md"""
	Show SPIROMICS X matrix $(@bind radio_Xspiro CheckBox())
	"""		
end	
end

# ╔═╡ af5ef391-b9bf-41e8-b539-5fa58f761765
begin 
if (@isdefined mXspiro)
	if radio_Xspiro
		mXspiro
	else
		md"The size of SPIROMICS X is $(size(mXspiro, 1)) x $(size(mXspiro, 2))."
	end
end
end

# ╔═╡ 1406d884-c1e2-4e84-b5d2-bb3bd0587e39
md"""
## Metabolites analysis
"""

# ╔═╡ ce038faa-ab1f-4686-b14d-2c7f8ba0a257
md"""
### Association at the metabolite level
"""

# ╔═╡ b180bc39-33ca-41e6-9240-f69ebdc1085c
begin
	
if (@isdefined mXcopd)
	###################
	# Identity Matrix #
	###################
	ZI = Matrix{Float64}(I(copd.m))
	
	copdZI = getcoeffs_mlm(copd.mY, mXcopd, ZI);
	spiroZI = getcoeffs_mlm(spiro.mY, mXspiro, Matrix{Float64}(I(spiro.m)));
 
		
	dfTstatsZIcopd = DataFrame(
		hcat(permutedims(copdZI.tstats),names(copd.dfMet)[2:end]),
		vcat(vPseudoFrmlNames_copd, ["CompID"])
		);
	
	dfTstatsZIcopd = leftjoin(
		dfTstatsZIcopd, 
		copd.dfRef[:,[:CompID, :SuperClassID , :SubClassID]], on = :CompID
		);

	dfTstatsZIspiro = DataFrame(
		hcat(permutedims(spiroZI.tstats),names(spiro.dfMet)[2:end]),
		vcat(vPseudoFrmlNames_spiro, ["CompID"])
		);
	
	dfTstatsZIspiro = leftjoin(
		dfTstatsZIspiro, 
		spiro.dfRef[:,[:CompID, :SuperClassID , :SubClassID]], on = :CompID
		);

	# need to convert for visualization
	for i in Symbol.(vPseudoFrmlNames_copd[idx_covar_copd])
		# dfTstatsZI[:, i] = float.(vec(dfTstatsZI[:, i]))
		dfTstatsZIcopd[!,i] = convert.(Float64, dfTstatsZIcopd[!,i])
	end
	
	# need to convert for visualization
	for i in Symbol.(vPseudoFrmlNames_spiro[idx_covar_spiro])
		dfTstatsZIspiro[!,i] = convert.(Float64, dfTstatsZIspiro[!,i])
	end


	
	md"""
	Show COPDGene Z matrix $(@bind radio_ZI CheckBox())
	"""		
end
	
end

# ╔═╡ 440f93f3-6c51-4dea-bd69-a6387bdfeecc
begin 
if (@isdefined copd_mX)
	if radio_ZI
		ZI
	else
		md"Z matrix is an identity matrix."
	end
end
end

# ╔═╡ 4be72887-8c6d-4812-80f8-6c75e55d73ee
if @isdefined vCovarNames
	@bind xCovarFig Select(vFrmlNames_copd[idx_covar_copd], default = "Sex: Female")
end

# ╔═╡ 87f8fc10-c6f0-4487-9394-d22e3b480258
begin
	if @isdefined dfTstatsZIcopd

		# COPDGene
		nameCovarFig = fix_covar_name(xCovarFig)
		gdf = groupby(dfTstatsZIcopd, :SuperClassID);
		dfMeanTst_copd = DataFrames.combine(gdf, Symbol(nameCovarFig) => mean => Symbol(nameCovarFig)) 
		sort!(dfMeanTst_copd, :SuperClassID);
	
		# var2vis = Symbol("Sex")
		p_copd = eval(Meta.parse("@df dfTstatsZIcopd dotplot(string.(:SuperClassID), :$(nameCovarFig), legend = false, markersize = 4)"))
		eval(Meta.parse("@df dfMeanTst_copd scatter!(string.(:SuperClassID), :$(nameCovarFig), legend = false, color = :orange)"))
		hline!([0], color= :red, 
			label = "",
			xlabel = "Super class",
			ylabel = string("T-statistics ", xCovarFig),
			title = "T-statistics per class - COPDGene",
			titlefontsize = mytitlefontsize,
			fontfamily = myfont, grid = false,
		)

		# SPIROMICS
		gdf = groupby(dfTstatsZIspiro, :SuperClassID);
		dfMeanTst_spiro = DataFrames.combine(gdf, Symbol(nameCovarFig) => mean => Symbol(nameCovarFig)) 
		sort!(dfMeanTst_spiro, :SuperClassID);
	
		# var2vis = Symbol("Sex")
		p_spiro = eval(Meta.parse("@df dfTstatsZIspiro dotplot(string.(:SuperClassID), :$(nameCovarFig), legend = false)"))
		eval(Meta.parse("@df dfMeanTst_spiro scatter!(string.(:SuperClassID), :$(nameCovarFig), legend = false, color = :orange)"))
		hline!([0], color= :red, 
			label = "",
			xlabel = "Super class",
			ylabel = string("T-statistics ", xCovarFig),
			title = "T-statistics per class - SPIROMICS",
			titlefontsize = mytitlefontsize,
			fontfamily = myfont, grid = false,
			size = (500, 700)
		)

		plot(p_copd, p_spiro, layout = @layout [a; b])
		
	end
end

# ╔═╡ f5517a21-4885-447c-800f-a065e387e935
md"""
### Association at the super class level
"""

# ╔═╡ cb63e350-0f68-457c-8098-1d57000e40c1
begin
	
if (@isdefined mXcopd)
	###############
	# Super Class #
	###############

	# COPDGene
	levelsSup_copd = sort(unique(copd.dfRef.SuperClassID));
	# Generate Z matrix
	mZsup_copd = modelmatrix(@formula(y ~ 0 + SuperClassID).rhs, 
						copd.dfRef, 
						hints = Dict(:SuperClassID => StatsModels.FullDummyCoding()));
	
	# true indicates correct design matrix
	# levelsSup[(mapslices(x ->findall(x .== 1) , mZsup, dims = [2]))[:]] == dfRef.SuperClassID
	copdZsp = getcoeffs_mlm(copd.mY, mXcopd, mZsup_copd);	

	# SPIROMICS
	levelsSup_spiro = sort(unique(spiro.dfRef.SuperClassID));
	# Generate Z matrix
	mZsup_spiro = modelmatrix(@formula(y ~ 0 + SuperClassID).rhs, 
						spiro.dfRef, 
						hints = Dict(:SuperClassID => StatsModels.FullDummyCoding()));
	
	# true indicates correct design matrix
	# levelsSup[(mapslices(x ->findall(x .== 1) , mZsup, dims = [2]))[:]] == dfRef.SuperClassID
	spiroZsp = getcoeffs_mlm(spiro.mY, mXspiro, mZsup_spiro);
	
	md"""
	Show Z matrix $(@bind radio_Zsp CheckBox())
	"""		
end
	
end


# ╔═╡ a0b8461a-493f-4ee2-921b-7cc3c3b30ce4
begin 
if (@isdefined mXcopd)
	if radio_Zsp
		DataFrame(hcat(Int.(mZsup_copd[:,1:end]), copd.dfRef.SuperClassID[:]), vcat(levelsSup_copd, ["Class"]))
	else
		md"Z matrix design is based on super classes."
	end
end
end

# ╔═╡ b9a70c70-a560-419a-b86b-df252df9ef85
if @isdefined vFrmlNames
	@bind xCovarFig_sp Select(vFrmlNames_copd[idx_covar_copd], default = "Sex: Female")
end

# ╔═╡ a321e2ce-0307-4baa-9d9e-821236a84f36
begin
	# if @isdefined TstatZsp
	# 	namesX = xCovarFig
	# 	idxCovar = findall(vCovarNames.==xCovarFig)
	# 	namesZsup = string.(permutedims(levelsSup[1:end]))
	# 	plot(transpose(TstatZsp[idxCovar,1:end]), # [:, 1:2]
	# 	    label= namesX, xlabel= "Super Pathway", ylabel = "T-statistics",
	# 	    legend =:outerright, lw = 2, marker = :circle, grid = false,
	# 	    xticks = (collect(1:size(mZsup)[2]), namesZsup),xrotation = 45,
	# 		size = (800,550),
	# 		# size = (650,550),
	# 	    foreground_color_legend = nothing, # remove box of legend   
	# 	    bottom_margin = 10mm, format = :svg, 
	# 	    title = string("T-statistics of Coefficients Estimates") )
	# 	hline!([2], color= :red, label = "")
	# 	hline!([-2], color= :red, label = "")
		
		# namesX = permutedims(vCovarNames[(vCovarNames[1] == "Intercept" ? 2 : 1):end])
		# namesZ = string.(permutedims(levelsSup[1:end]))
		# plot(transpose(TstatZsp[:,1:end])[:,2:end], # [:, 1:2]
		#     label= namesX, xlabel= "Super Pathway", ylabel = "T-statistics",
		#     legend =:outerright, lw = 2, marker = :circle, grid = false,
		#     xticks = (collect(1:size(mZsup)[2]), namesZ),xrotation = 45,
		# 	size = (800,550),
		#     foreground_color_legend = nothing, # remove box of legend   
		#     bottom_margin = 10mm, format = :svg,
		#     title = string("T-statistics of Coefficients Estimates") )
		# hline!([2], color= :red, label = "")
		# hline!([-2], color= :red, label = "")
	end
# end

# ╔═╡ 923b38bb-9f09-4591-a229-6552fd756307
begin
	if @isdefined copd
		
		# idxCovarsup = findall(vPseudoFrmlNames_copd[1:end-1] .== fix_covar_name(xCovarFig_sp))

		idxCovarsup_copd = findall(
			vPseudoFrmlNames_copd .== fix_covar_name(xCovarFig_sp))
		idxCovarsup_spiro = findall(
			vPseudoFrmlNames_spiro .== fix_covar_name(xCovarFig_sp))

		
		idx2rmv_copd = findall(levelsSup_copd .== "PAR" .|| 
							levelsSup_copd .== "NA")
		psp_copd = confidenceplot(
			vec(permutedims(copdZsp.coef[idxCovarsup_copd,Not(idx2rmv_copd)])), 
			# levelsSup_copd[Not(idx2rmv_copd)],
			sort(unique(copd.dfRef.SuperPathway))[Not(idx2rmv_copd)],
			vec((copdZsp.ci[idxCovarsup_copd,Not(idx2rmv_copd)])),
			xlabel = xCovarFig_sp*" Effect Size", legend = false,
			fontfamily = myfont,
			title = "COPDGene",
			titlefontsize = mytitlefontsize,
			right_margin = -5mm,
			xlim = (-.12,.15),
			size = (800, 400),
		)

		
		namesZsup_spiro = string.(permutedims(levelsSup_spiro[1:end]))
		idx2rmv_spiro = findall(levelsSup_spiro .== "PAR" .|| 
							levelsSup_spiro .== "NA")
		psp_spiro = confidenceplot(
			vec(permutedims(spiroZsp.coef[idxCovarsup_spiro,Not(idx2rmv_spiro)])),
			levelsSup_spiro[Not(idx2rmv_spiro)],
		# 	vec(namesZsup_spiro),
			vec((spiroZsp.ci[idxCovarsup_spiro,Not(idx2rmv_spiro)])),
			xlabel = xCovarFig_sp*" Effect Size", legend = false,
			fontfamily = myfont, 
			y_foreground_color_text = :white,
			title = "SPIROMICS",
			titlefontsize = mytitlefontsize,
			# right_margin = 10mm,
			left_margin = -5mm,
			bottom_margin = 5mm,
			xlim = (-.12,.15),
			size = (800, 400),
		)

		plot(psp_copd, psp_spiro, fontfamily = myfont, layout = @layout [a b])
		savefig(string(
			"../images/copd_mlmCI_",
			replace(xCovarFig_sp, " "=> "_", ":"=> "_"),
			"_",
			"super_class",
			".svg")
		)
		plot(psp_copd, psp_spiro, fontfamily = myfont, layout = @layout [a b])

		
	end
end

# ╔═╡ 660f3468-00ba-4938-83a6-9bdbb20d4795
begin 
	
	# ncovar = length(vFrmlNames_copd)-1
	critical_level_sp = 1.96;
	
	diffcoef_sp = spiroZsp.coef[idx_covar_spiro, Not(idx2rmv_spiro)] .- 
				copdZsp.coef[idx_covar_copd, Not(idx2rmv_copd)];

	avgcoef_sp = (spiroZsp.coef[idx_covar_spiro, Not(idx2rmv_spiro)] .+ 
			copdZsp.coef[idx_covar_copd, Not(idx2rmv_copd)])./2;
	

	SEcopd_sp = sqrt.(copdZsp.var[idx_covar_copd, Not(idx2rmv_copd)]);
	SEspiro_sp = sqrt.(spiroZsp.var[idx_covar_spiro, Not(idx2rmv_spiro)]);
	SEdiff_sp = sqrt.(SEcopd_sp.^2 .+ SEspiro_sp.^2);

	CIdiff_sp = SEdiff_sp.*critical_level_sp; 

	idx_avg_diff_sup= findall(idx_covar_copd .== idxCovarsup_copd)

	p1_sp = confidenceplot(
		# vec(permutedims(avgcoef_mdl[idxCovarmdl, df.idx])),
		vec(permutedims(avgcoef_sp[idx_avg_diff_sup, :])),
		# levelsSup_spiro[Not(idx2rmv_spiro)],
		sort(unique(copd.dfRef.SuperPathway))[Not(idx2rmv_copd)],
		vec((CIdiff_sp[idx_avg_diff_sup, :])),
		
		xlabel = xCovarFig_sp*" Effect Size", legend = false,
		fontfamily = myfont, 
		title = "Average",
		titlefontsize = mytitlefontsize,
		right_margin = -5mm,
		xlim = (-.12,.15) ,
		size = (800, 400),	
	)
	
	p2_sp = confidenceplot(
		vec(permutedims(diffcoef_sp[idx_avg_diff_sup,:])),
		levelsSup_spiro[Not(idx2rmv_spiro)],
		vec((CIdiff_sp[idx_avg_diff_sup, :])),
		xlabel = xCovarFig_sp*" Effect Size", legend = false,
		fontfamily = myfont, y_foreground_color_text = :white,
		title = "Difference",
		titlefontsize = mytitlefontsize,
		# right_margin = 10mm,
		left_margin = -5mm,
		bottom_margin = 5mm,
		xlim = (-.12,.15),
		size = (800, 400),
	)

	plot(p1_sp, p2_sp, layout = @layout [a b])

	savefig(string(
		"../images/copd_mlmCI_",
		replace(xCovarFig_sp, " "=> "_", ":"=> "_"),
		"_avg_diff_",
		"super_class",
		".svg")
	)
	plot(p1_sp, p2_sp, layout = @layout [a b])
	
end

# ╔═╡ d427bc34-94a4-48bb-b880-d604a67a8d4a
begin
if (@isdefined copd2) && (xCovariates != ["Intercept"])
	catTstatZsp = zeros(size(copdZsp.tstats,1), size(copdZsp.tstats,2));
	catTstatZsp[findall(abs.(copdZsp.tstats) .<2)].= 0;
	catTstatZsp[findall((copdZsp.tstats .>=2) .&& (copdZsp.tstats .<3))].= 1;
	catTstatZsp[findall((copdZsp.tstats .<=-2) .&& (copdZsp.tstats .>-3))].= -1;
	catTstatZsp[findall(copdZsp.tstats .>=3)].= 2;
	catTstatZsp[findall(copdZsp.tstats .<=-3)].= -2;
	catTstatZsp
	
	# myscheme = [RGB{Float64}(0.792, 0, 0.125),
 #            RGB{Float64}(0.957, 0.647, 0.51),
 #            RGB{Float64}(0.5, 0.5, 0.5),    
 #            RGB{Float64}(0.557, 0.773, 0.871),
 #            RGB{Float64}(0.02, 0.443, 0.69)];

	myscheme = [RGB{Float64}(0.471, 0.161, 0.518),
            RGB{Float64}(0.694, 0.549, 0.761),
            RGB{Float64}(0.5, 0.5, 0.5),    
            RGB{Float64}(0.502, 0.745, 0.49),
            RGB{Float64}(0.114, 0.467, 0.216)];
	# labels
	namesTstsX = vFrmlNames
	namesXtbl = reshape(namesTstsX[2:end], 1, length(namesTstsX)-1)
	# namesTstsX = vCovarNames
	# namesXtbl = reshape(namesTstsX[2:end], 1, length(namesTstsX)-1)
	clrbarticks = ["T-stats ≤ -3", "-3 < T-stats ≤ -2", "-2 < T-stats < 2", "2 ≤ T-stats < 3", "3 ≤ T-stats"];
	n_cat = length(clrbarticks)
	yt = range(0,1,n_cat+1)[1:n_cat] .+ 0.5/n_cat
	# myfont = "Computer Modern" 

	l = @layout [ a{0.50w} [b{0.1h}; c{0.2w}] ];

	mycolors =ColorScheme(myscheme)
	colors = cgrad(mycolors, 5, categorical = true);

	# PLOT MAIN HEATMAP
	p1 = mlmheatmap(permutedims(catTstatZsp[2:end,:]),
		permutedims(copdZsp.tstats[2:end,:]), 
	    xticks = (collect(1:size(mZsup_copd)[1]), namesXtbl), xrotation = 40,
	    yticks = (collect(1:size(mZsup_copd)[2]), levelsSup_copd),
		tickfontfamily = myfont ,
	    color = colors, cbar = false,
		clims = (-3, 3),
	    annotationargs = (10, "Arial", :lightgrey), 
	    linecolor = :white , linewidth = 2);

	# PLOT GAP 
	p2 = plot([NaN], lims=(0,1), framestyle=:none, legend=false);

	# ANNOTATE COLOR BAR TITLE
	annotate!(0.5, -3.02, text("T-statistics", 10, myfont))
	
	# PLOT HEATMAP COLOR BAR
	xx = range(0,1,100)
	zz = zero(xx)' .+ xx
	p3 = Plots.heatmap(xx, xx, zz, ticks=false, ratio=3, legend=false, 
	            fc=colors, lims=(0,1), framestyle=:box, right_margin=20mm);
	
	# ANNOTATE COLOR BAR AXIS
	[annotate!(2.15, yi, text(ti, 7, myfont)) for (yi,ti) in zip(yt,clrbarticks)]
	
	# PLOT MLMHEATMAP
	plot(p1, p2, p3, layout=l, margins=4mm)
end
end

# ╔═╡ 0f942bb6-ea42-4988-93f4-6afa3874f91a
if @isdefined(TstatZsp)
	mlmheatmap(permutedims(TstatZsp[2:end, :]), permutedims(TstatZsp[2:end,:]), 
		    xticks = (collect(1:size(mZsup)[1]), namesXtbl), xrotation = 25,
		    yticks = (collect(1:size(mZsup)[2]), levelsSup), tickfontfamily = myfont ,
		    color = cgrad(:bluesreds,[0.1, 0.3, 0.7, 0.9], alpha = 0.8), cbar = true,
		 	clims = (-3, 3), colorbar_title = "T-statistics",
		    annotationargs = (10, "Arial", :lightgrey), 
		    linecolor = :white , linewidth = 2, margins=8mm)
end

# ╔═╡ 202b9efe-a41a-4ade-929f-c719a4a17abd
md"""
### Association at the sub class level
"""

# ╔═╡ 7d3574f1-a6ff-4629-8fe0-b40ba4de22a8
begin
	
if (@isdefined mXcopd)
	#############
	# Sub Class #
	#############
	
	# COPDGene
	levelsSub_copd = sort(unique(copd.dfRef.SubClassID));
	# Generate Z matrix
	mZsub_copd = modelmatrix(@formula(y ~ 0 + SubClassID).rhs, 
						copd.dfRef, 
						hints = Dict(:SubClassID => StatsModels.FullDummyCoding()));
	
	# true indicates correct design matrix
	# levelsSub[(mapslices(x ->findall(x .== 1) , mZsub, dims = [2]))[:]] == dfRef.SubClassID
	copdZsb = getcoeffs_mlm(copd.mY, mXcopd, mZsub_copd);	

	# SPIROMICS
	levelsSub_spiro = sort(unique(spiro.dfRef.SubClassID));
	# Generate Z matrix
	mZsub_spiro = modelmatrix(@formula(y ~ 0 + SubClassID).rhs, 
						spiro.dfRef, 
						hints = Dict(:SubClassID => StatsModels.FullDummyCoding()));
	
	# true indicates correct design matrix
	# levelsSup[(mapslices(x ->findall(x .== 1) , mZsup, dims = [2]))[:]] == dfRef.SuperClassID
	spiroZsb = getcoeffs_mlm(spiro.mY, mXspiro, mZsub_spiro);

	############################################
	# Adjust idx to compare similar sub class #
	############################################

	# create copd dataframe with the unique SubClassID
	dfcopd = select(copd.dfRef, 
			:SubPathway,
			:SubClassID =>:SubClassIDcopd,
			:SuperClassID);
	dfspiro = select(spiro.dfRef, 
			:SubPathway,
			:SubClassID => :SubClassIDspiro);

	# select subpathway with more or equal 5 counts
	dfcopd2 = filter(row -> row.nrow > 4,  DataFrames.combine(groupby(dfcopd[:, :], [:SubPathway]), nrow))
	
	dfspiro2 = filter(row -> row.nrow > 4,  DataFrames.combine(groupby(dfspiro[:, :], [:SubPathway]), nrow))


	
	# get common sub pathway
	vintersub = intersect(dfcopd2.SubPathway, dfspiro2.SubPathway)
	
	# # select indices for common sub pathway
	# idx_copd_intersub = findall(x -> [x] ⊆ vintersub, copd.dfRef.SubPathway);
	# idx_spiro_intersub = findall(x -> [x] ⊆ vintersub, spiro.dfRef.SubPathway);
	
	# keep unique  ID
	dfcopd = sort(
			DataFrames.combine(groupby(dfcopd[:, :], [:SubClassIDcopd]), last), :SubClassIDcopd
			)  
	insertcols!(dfcopd, 1, :idxcopd => 1:size(dfcopd, 1))
	dfspiro = sort(
			DataFrames.combine(groupby(dfspiro[:, :], [:SubClassIDspiro]), last), :SubClassIDspiro
			)  
	insertcols!(dfspiro, 1, :idxspiro => 1:size(dfspiro, 1))

	# select indices for common sub pathway
	idx_copd_intersub = findall(x -> [x] ⊆ vintersub, dfcopd.SubPathway);
	idx_spiro_intersub = findall(x -> [x] ⊆ vintersub, dfspiro.SubPathway);
	
	
	dfcopd = dfcopd[idx_copd_intersub, :]
	dfspiro = dfspiro[idx_spiro_intersub, :]

	# join both copd and spiro
	df_spiro_copd = leftjoin(dfspiro, dfcopd, on = :SubPathway);
	
	#############################################################################
	
	md"""
	Show COPDGene Z matrix for $(@bind radio_Zsb CheckBox())
	"""		
end
	
end


# ╔═╡ e09bd194-ccb2-4448-8e05-1816dc261e31
begin 
if (@isdefined mXcopd)
	if radio_Zsb
		DataFrame(hcat(Int.(mZsub_copd[:,1:end]), copd.dfRef.SubClassID[:]), vcat(levelsSub_copd, ["Class"]))
	else
		md"Z matrix design is based on sub-classes."
	end
end
end

# ╔═╡ f91ab58d-0808-4558-824d-6f348316c1b4
if @isdefined vFrmlNames_copd
	@bind xCovarFig_sb Select(vFrmlNames_copd[idx_covar_copd], default = "Sex: Female")
end

# ╔═╡ e76052ae-be3b-4459-be18-6c4ecc903093
begin 
	if @isdefined copdZsb
		idxCovarsub_copd = findall(
			vPseudoFrmlNames_copd .== fix_covar_name(xCovarFig_sb))
		idxCovarsub_spiro = findall(
			vPseudoFrmlNames_spiro .== fix_covar_name(xCovarFig_sb))

		psb_copd = confidenceplot(
			vec(permutedims(copdZsb.coef[idxCovarsub_copd, df_spiro_copd.idxcopd])), 
			levelsSub_copd[df_spiro_copd.idxcopd],
			vec((copdZsb.ci[idxCovarsub_copd, df_spiro_copd.idxcopd])),
			xlabel = xCovarFig_sb*" Effect Size", legend = false,
			fontfamily = myfont,
			title = "COPDGene", 
			size=(800,700), 
			)
		
		psb_spiro = confidenceplot(
			vec(permutedims(spiroZsb.coef[idxCovarsub_spiro, df_spiro_copd.idxspiro])), 
			# levelsSub_spiro[idx_spiro_intersub],
			levelsSub_copd[df_spiro_copd.idxcopd],
			vec((spiroZsb.ci[idxCovarsub_spiro, df_spiro_copd.idxspiro])),
			xlabel = xCovarFig_sb*" Effect Size", legend = false,
			fontfamily = myfont,
			title = "SPIROMICS",
			size=(800,700), 
			)

		plot(psb_copd, psb_spiro, layout = @layout [a b])
		
		
		savefig(string(
			"../images/copd_mlmCI_",
			replace(xCovarFig_sb, " "=> "_", ":"=> "_"),
			"_",
			"sub_class",
			".svg")
		)
		plot(psb_copd, psb_spiro, layout = @layout [a b])
						
	end

end

# ╔═╡ 7d6c2a28-4f85-4939-a7c7-f5aeab385d7e
begin 
	
	critical_level_sb = 1.96;
	# diffcoef_sb = spiroZsb.coef[1:ncovar, df_spiro_copd.idxspiro] .- 
	# 			copdZsb.coef[1:ncovar, df_spiro_copd.idxcopd];
	
	
	diffcoef_sb = spiroZsb.coef[idx_covar_spiro, df_spiro_copd.idxspiro] .- 
				copdZsb.coef[idx_covar_copd, df_spiro_copd.idxcopd];

	avgcoef_sb = (spiroZsb.coef[idx_covar_spiro, df_spiro_copd.idxspiro] .+ 
			copdZsb.coef[idx_covar_copd, df_spiro_copd.idxcopd])./2;
	

	SEcopd_sb = sqrt.(copdZsb.var[idx_covar_copd, df_spiro_copd.idxcopd]);
	SEspiro_sb = sqrt.(spiroZsb.var[idx_covar_spiro, df_spiro_copd.idxspiro]);
	SEdiff_sb = sqrt.(SEcopd_sb.^2 .+ SEspiro_sb.^2);

	CIdiff_sb = SEdiff_sb.*critical_level_sb; 

	idx_avg_diff_sub= findall(idx_covar_copd .== idxCovarsub_copd)
	
	p1_sb = confidenceplot(
		# vec(permutedims(avgcoef_mdl[idxCovarmdl, df.idx])),
		vec(permutedims(avgcoef_sb[idx_avg_diff_sub, :])),
		levelsSub_copd[df_spiro_copd.idxcopd],
		vec((CIdiff_sb[idx_avg_diff_sub, :])),	
		xlabel = xCovarFig_sb*" Effect Size", legend = false,
		fontfamily = myfont, 
		title = "Average",
		titlefontsize = mytitlefontsize,
		# xlim = (-.3,.35) ,
		size = (800, 700),	
	)

	p2_sb = confidenceplot(
		vec(permutedims(diffcoef_sb[idx_avg_diff_sub,:])),
		levelsSub_copd[df_spiro_copd.idxcopd],
		vec((CIdiff_sb[idx_avg_diff_sub, :])),
		xlabel = xCovarFig_sb*" Effect Size", legend = false,
		fontfamily = myfont, y_foreground_color_text = :white, 
		title = "Difference",
		titlefontsize = mytitlefontsize,
		size = (800, 700),	
	)
	
	plot(p1_sb, p2_sb, layout = @layout [a b])

	savefig(string(
		"../images/copd_mlmCI_",
		replace(xCovarFig_sb, " "=> "_", ":"=> "_"),
		"_avg_diff_",
		"sub_class",
		".svg")
	)
	plot(p1_sb, p2_sb, layout = @layout [a b])

end

# ╔═╡ 0c94e2c0-3e65-47a5-9093-822ba610e9f2
begin 
	df = sort(
		DataFrame(
			val = vec(permutedims(avgcoef_sb[idx_avg_diff_sub,:])),
			idx = 1:size(avgcoef_sb , 2),
			SubPathway = string.(df_spiro_copd.SubPathway),
			idSubPathway = string.(df_spiro_copd.SubClassIDcopd),
		), 
		:val
	)

	
	p_avg_sb = 
	confidenceplot(
		vec(permutedims(avgcoef_sb[idx_avg_diff_sub, df.idx])),
		# df_spiro_copd.SubPathway,
		levelsSub_copd[df_spiro_copd.idxcopd][df.idx],
		vec((CIdiff_sb[idx_avg_diff_sub, df.idx])),
		xlabel = "COPDGene and SPIROMICS Average "*xCovarFig_sb*" Effect Size", 
		legend = false,
		fontfamily = myfont,
		titlefontsize = mytitlefontsize,
		# title = "COPDGene and SPIROMICS Average Effect Size",
		# xlim = (-.32,.42) ,
		size = (500, 700),
	)

	savefig(string(
		"../images/copd_mlmCI_average_",
		replace(xCovarFig_sb, " "=> "_", ":"=> "_"),
		"_avg_sort_",
		"sub_class",
		".svg")
	)
	plot(p_avg_sb)
end

# ╔═╡ a65d4195-8580-4a6e-985d-1afa0d407f78
md"""
The followling tables shows the significant subpathway at a 95% confidence interval:
"""

# ╔═╡ c8f666d6-953c-4d53-99aa-6ec72fd82669
begin
	xcopd = vec(permutedims(avgcoef_sb[idx_avg_diff_sub, df.idx]));
	εcopd =  vec((CIdiff_sb[idx_avg_diff_sub, df.idx]));
	vCritical = (xcopd+εcopd).*(xcopd-εcopd) .> 0
	idxCritical = findall(vCritical)
	
	if isempty(idxCritical)
		md"""
		No significant sub pathway at a 95% confidence interval.
		"""	
	else
		df_avg_sub_critical = copy(df[idxCritical, [:idSubPathway, :SubPathway]])
		df_avg_sub_critical.Effect = round.(xcopd[idxCritical], digits = 3);
		df_avg_sub_critical.CI = string.(
									"[",
									round.((xcopd.-εcopd)[idxCritical], digits = 3),
									", ",
									round.((xcopd.+εcopd)[idxCritical], digits = 2),
									"]"
								 );
		
		latexify(df_avg_sub_critical;
				env=:mdtable, latex=false)
		# to get the latex code :
		latexify(
			df_avg_sub_critical;
			env=:table, latex=false
		) |> println
		
	end
end

# ╔═╡ 1ee7e175-1b6d-4f08-8ec7-945394e36311
begin 
	subnames = String.(levelsSub_copd[df_spiro_copd.idxcopd])
	# levelsSub_copd[df_spiro_copd.idxcopd][df.idx];
	se = vec((CIdiff_sb[idx_avg_diff_sub, :]))./1.96;
		# vec((CIdiff_sb[idx_avg_diff_sub, df.idx]))./1.96;
	est = vec(permutedims(avgcoef_sb[idx_avg_diff_sub, :]));
		# vec(permutedims(avgcoef_sb[idx_avg_diff_sub, df.idx]));
	mytstats = est./se 
	pvalues = ccdf.(TDist(size(mXcopd, 1)-1), abs.(mytstats)).*2
	qvalues = pval2qval(pvalues)
	qvalues2 = pval2qval2(pvalues)

	idx_sub_sginif = findall(qvalues2 .<= 0.05)

	df_sub_signif = DataFrame(
		ID = subnames,
		name = df_spiro_copd.SubPathway,
		Tstats = mytstats,
		Pvalues = pvalues,
		FDR = qvalues2
	)
end

# ╔═╡ 2facc8b7-0300-4b31-bb51-e9c4ee2b7fd6
ccdf.(TDist(size(mXcopd, 1)-1), abs.([0.357726, 7.28])).*2

# ╔═╡ c610e158-1028-47e7-877f-7a6f7e863cb3
ccdf.(TDist(size(mXcopd, 1)-1), abs.(mytstats[1:10])).*2

# ╔═╡ 4cc3ecef-a047-4362-af4b-1e7e78564823
mytstats[1:10]

# ╔═╡ 4b0bebfc-3162-427f-8430-f86eaeaa0b3c
est

# ╔═╡ d8885ae3-162d-49ea-b3f5-1030d76f73a0
avgcoef_sb[idx_avg_diff_sub, :]

# ╔═╡ b9bc72e7-e539-4715-9c75-bcd9ada7182b
DataFrame(
		ID = subnames,
		name = (copy(df)).SubPathway,
		Pvalues = pvalues,
		FDR = qvalues2
	)

# ╔═╡ 6b89cc56-234f-4246-a200-fed293c7fffc
CIdiff_sb[idx_avg_diff_sub, :]

# ╔═╡ 97af3d9f-1ed3-4bed-a46e-1682429c6355
# est[54]/se[54]

ccdf.(TDist(size(mXcopd, 1)-1), abs.(7.28)).*2

# ╔═╡ da2a35a9-1dbd-4bd5-acbc-23f51ddc250e
String.(levelsSub_copd[df_spiro_copd.idxcopd])
	# εcopd

# ╔═╡ ddf1313f-ecef-4970-9894-3ef2c832a33a
df_spiro_copd

# ╔═╡ ea420bf4-8e1b-4633-8ac3-75fd7f73ef91
findall(qvalues .<= 0.05)

# ╔═╡ 38f8472e-5930-4ea8-ae28-73781055c08b
histogram(pvalues)

# ╔═╡ 29e5234b-0459-4531-9c5b-eff5a7eb8a67
histogram(qvalues)

# ╔═╡ 60ef862a-f67a-4ea4-ae6d-74a5d2acfe93
begin 
	# est = xcopd
	# lo = (xcopd.-εcopd)
	# up = (xcopd.+εcopd)

	# # se = (up .− lo)./(2*1.96)
	# # SEdiff_sb  = εcopd./1.96
	# tstats = avgcoef_sb./SEdiff_sb 
	# pvalues = ccdf.(TDist(length(tstats)-1), abs.(tstats)).*2
end

# ╔═╡ 37c86980-6241-4121-ba87-6a3d8dc5a1ca

# df_avg_sub_critical.CI
pvalues_2 = 
	ccdf.(TDist(length(tstats)-1),
	abs.(tstats)).*2


# ╔═╡ 5d92b5bd-fc46-49d1-b1fc-c13c00433265
? Tdist

# ╔═╡ df1598ed-d2bc-44eb-8c66-fbc57b262d41
if @isdefined(TstatZsb)
	mlmheatmap(permutedims(TstatZsb[2:end, :]), permutedims(TstatZsb[2:end,:]), 
		    xticks = (collect(1:size(mZsub)[1]), namesXtbl), xrotation = 25,
		    yticks = (collect(1:size(mZsub)[2]), levelsSub), tickfontfamily = myfont ,
		    color = cgrad(:bluesreds,[0.1, 0.3, 0.7, 0.9], alpha = 0.8), cbar = true,
		 	clims = (-3, 3), colorbar_title = "T-statistics",
		    annotationargs = (10, "Arial", :lightgrey), 
		    linecolor = :white , linewidth = 2, margins=8mm)
end

# ╔═╡ d161fce0-14dd-44db-ba83-a6f58f4d9bdf
begin
	if (@isdefined TstatZsb) 
		# namesX = permutedims(vCovarNames[(vCovarNames[1] == "Intercept" ? 2 : 1):end])
		# namesZsub = string.(permutedims(levelsSub[1:end]))
		# plot(transpose(TstatZsb[idxCovar,1:end]), # [:, 1:2]
		#     label= namesX, xlabel= "Sub Pathway", ylabel = "T-statistics",
		#     legend =:outerright, lw = 2, marker = :circle, grid = false,
		#     xticks = (collect(1:size(mZsub)[2]), namesZsub),xrotation = 45,
		# 	size = (800,550),
		# 	# size = (650,550),
			
		#     foreground_color_legend = nothing, # remove box of legend   
		#     bottom_margin = 10mm, format = :svg,
		#     title = string("T-statistics of Coefficients Estimates") )
		# hline!([2], color= :red, label = "")
		# hline!([-2], color= :red, label = "")
				
		
		# namesX = permutedims(vCovarNames[(vCovarNames[1] == "Intercept" ? 2 : 1):end])
		# namesZsub = string.(permutedims(levelsSub[1:end]))
		# plot(transpose(TstatZsb[:,1:end])[:,2:end], # [:, 1:2]
		#     label= namesX, xlabel= "Super Pathway", ylabel = "T-statistics",
		#     legend =:outerright, lw = 2, marker = :circle, grid = false,
		#     xticks = (collect(1:size(mZsub)[2]), namesZsub),xrotation = 45,
		# 	size = (800,550),
		#     foreground_color_legend = nothing, # remove box of legend   
		#     bottom_margin = 10mm, format = :svg,
		#     title = string("T-statistics of Coefficients Estimates") )
		# hline!([2], color= :red, label = "")
		# hline!([-2], color= :red, label = "")
	end
end

# ╔═╡ c090c950-08cf-4b3f-a06d-e96f51b1b52c
md"""
### Association at the Modules level
"""

# ╔═╡ c5b22684-0221-4d17-b6e2-0196f0053f87
begin 
	df_mdl = DataFrame(
	[
	"blue" "Acyl Carnitines, Fatty Acids (Dicarboxylate, Monohydroxy, Long chain, Medium chain), Endocannabinoids, Nucleotides";
	"red" "Ceramides, Sphingomyelins";
	"turquoise" "Xenobiotics, Amino Acids (Tryptophan metabolism, Glutamate metabolism, Histidine metabolism, Branched Chain Amino Acids, Glycine, Serine and Threonine Metabolism, Methionine, Cysteine, SAM and Taurine Metabolism, Polyamine Metabolism, Urea cycle; Arginine and Proline Metabolism), TCA cycle metabolites";
	"brown" "Amino Acids (Gamma-glutamyl Amino Acid, Glutamate Metabolism, Branched Chain Amino Acids, Urea cycle; Arginine and Proline Metabolism, Lysine Metabolism, Methionine, Cysteine, SAM and Taurine Metabolism, Phenylalanine Metabolism), Bile Acids, Acyl Cholines, Lysophospholipids";
	"yellow" "Xenobiotics (Benzoates, Xanthines, Nutritional)";
	"green" "Lysophospholipids, Phosphatidylcholines (PC), Phosphatidylinositols (PI), Plasmalogens";
	"magenta" "Sterioids (Androgenic, Pregnenolone, Corticosteroids, Progestin)";
	"black" "Diacylglycerols, Phosphatidylethanolamines (PE), Acyl Carnitines";
	"greenyellow" "Cofactors and Vitamins";
	"purple" "Acetylated peptides, Xenobiotics (Benzoates), Secondary Bile Acids";
	"pink" "Xenobiotics (Chemicals), Dipeptides, Hemoglobin and Porphyrin Metabolites"
	], ["Module", "Metabolite Classes"]
	);

	# to get the latex code:
	# latexify(df_mdl;
	# 			env=:table, latex=false) |> println
	latexify(df_mdl;
				env=:mdtable, latex=false)
end

# ╔═╡ 08836fbc-577e-4793-86ab-cdae1bf08f56
begin 
	nameModules =  ["pink", "purple", "greenyellow", "black", "magenta", "green",
		"yellow", "brown", "turquoise", "red", "blue"]

	spiroZmdl, mZmdl_spiro = get_modules_coefs(spiro, mXspiro);
	copdZmdl, mZmdl_copd = get_modules_coefs(copd, mXcopd);
	
	md"""
	Show COPDGene Z matrix $(@bind radio_Zmdl CheckBox())
	"""		
end

# ╔═╡ f572cc42-e003-40a2-be0b-11b52a6cc946
begin 
if (@isdefined mXcopd)
	if radio_Zmdl
		DataFrame(Int.(mZmdl_copd[:,1:end]), nameModules)
	else
		md"Z matrix design is based on modules."
	end
end
end

# ╔═╡ 77171c73-5281-4c72-92cd-459faedece45
if @isdefined vFrmlNames
	@bind xCovarFig_mdl Select(vFrmlNames_copd[idx_covar_copd], default = "Sex: Female")
end

# ╔═╡ 287da42a-4294-4942-8f33-7b478cddfbfd
begin 
	if @isdefined copdZmdl
		# idxCovarmdl = findall(vPseudoFrmlNames_copd .== fix_covar_name(xCovarFig_mdl))

		idxCovarmdl_copd = findall(
			vPseudoFrmlNames_copd .== fix_covar_name(xCovarFig_mdl))
		idxCovarmdl_spiro = findall(
			vPseudoFrmlNames_spiro .== fix_covar_name(xCovarFig_mdl))
		
		pmdl_copd = confidenceplot(
			vec(permutedims(copdZmdl.coef[idxCovarmdl_copd, :])), 
			nameModules,
			vec((copdZmdl.ci[idxCovarmdl_copd, :])),
			xlabel = xCovarFig_mdl*" Effect Size", legend = false,
			fontfamily = myfont,
			title = "COPDGene", 
			titlefontsize = mytitlefontsize,
			right_margin = -5mm,
			# size=(800,700), 
			)
		
		pmdl_spiro = confidenceplot(
			vec(permutedims(spiroZmdl.coef[idxCovarmdl_spiro, :])), 
			nameModules,
			vec((spiroZmdl.ci[idxCovarmdl_spiro, :])),
			xlabel = xCovarFig_mdl*" Effect Size", legend = false,
			fontfamily = myfont, y_foreground_color_text = :white, 
			title = "SPIROMICS",
			titlefontsize = mytitlefontsize,
			left_margin = -5mm,
			# size=(800,700), 
			)

		plot(pmdl_copd, pmdl_spiro, layout = @layout [a b])

		savefig(string(
			"../images/copd_mlmCI_",
			replace(xCovarFig_sb, " "=> "_", ":"=> "_"),
			"_",
			"modules",
			".svg")
		)
		plot(pmdl_copd, pmdl_spiro, layout = @layout [a b])



		
	end

end

# ╔═╡ 44a53807-3ff5-423b-9745-18be61fc2af3
begin 
	critical_level_mdl = 1.96;
	ws = size(spiro.dfInd, 1)/(size(spiro.dfInd, 1)+size(copd.dfInd, 1));
	wc = size(copd.dfInd, 1)/(size(spiro.dfInd, 1)+size(copd.dfInd, 1));

	diffcoef_mdl = spiroZmdl.coef[idx_covar_spiro, :] .- 
				copdZmdl.coef[idx_covar_copd, :];

	avgcoef_mdl = (spiroZmdl.coef[idx_covar_spiro, :] .+ 
				copdZmdl.coef[idx_covar_copd, :])./2;
	
	# avgcoef_mdl = ws.*spiroZmdl.coef[idx_covar_spiro, :] .+ 
	# 			wc.*copdZmdl.coef[idx_covar_copd, :];
	

	SEcopd_mdl = sqrt.(copdZmdl.var[idx_covar_copd, :]);
	SEspiro_mdl = sqrt.(spiroZmdl.var[idx_covar_spiro, :]);
	SEdiff_mdl = sqrt.(SEcopd_mdl.^2 .+ SEspiro_mdl.^2);

	CIdiff_mdl = SEdiff_mdl.*critical_level_mdl; 

	idx_avg_diff_mdl= findall(idx_covar_copd .== idxCovarmdl_copd)
	
	# df = sort(DataFrame(val = vec(permutedims(diffcoef_mdl[idx_avg_diff_mdl, :])),
	# idx = 1:size(diffcoef_mdl , 2)), :val)

	p1_mdl = confidenceplot(
		# vec(permutedims(avgcoef_mdl[idxCovarmdl, df.idx])),
		vec(permutedims(avgcoef_mdl[idx_avg_diff_mdl, :])),
		nameModules[:],
		vec((CIdiff_mdl[idx_avg_diff_mdl, :])),
		
		xlabel = xCovarFig_mdl*" Effect Size", legend = false,
		fontfamily = myfont,
		title = "Average", 
		titlefontsize = mytitlefontsize,
		xlim = (-.3,.35) ,
		size = (500, 600),	
	)

	p2_mdl = confidenceplot(
		# vec(permutedims(diffcoef_mdl[idxCovarmdl, df.idx])),
		vec(permutedims(diffcoef_mdl[idx_avg_diff_mdl, :])),
		nameModules[:],
		vec((CIdiff_mdl[idx_avg_diff_mdl, :])),
		
		xlabel = xCovarFig_mdl*" Effect Size", legend = false,

		# xtick = -0.08:0.04:0.08,
		
		fontfamily = myfont,y_foreground_color_text = :white, 
		title = "Difference", 
		titlefontsize = mytitlefontsize,
		xlim = (-.3,.35) ,
		size = (500, 600),	
	)
	
	plot(p1_mdl, p2_mdl, layout = @layout [a b])
	savefig(string(
			"../images/copd_mlmCI_",
			replace(xCovarFig_sb, " "=> "_", ":"=> "_"),
			"_avg_diff_",
			"modules",
			".svg")
		)
	plot(p1_mdl, p2_mdl, layout = @layout [a b])
	
end

# ╔═╡ afad2cfb-d394-43fb-b434-f4de6bd4dff6
begin
# plot(pmdl_copd, pmdl_spiro, p1_mdl, p2_mdl, layout = @layout [a b;c d], size = ())
end

# ╔═╡ f1c0cd37-362b-4fe6-b8e3-37ba3886ce2d
# let 
# 	critical_level = 1.96;
# 	diffcoef_mdl = spiroZmdl.coef[:, :] .- 
# 				copdZmdl.coef[:, :];

# 	avgcoef_mdl = (spiroZmdl.coef[:, :] .+ 
# 				copdZmdl.coef[:, :])./2;
	

# 	SEcopd_mdl = sqrt.(copdZmdl.var[:, :]);
# 	SEspiro_mdl = sqrt.(spiroZmdl.var[:, :]);
# 	SEdiff_mdl = sqrt.(SEcopd_mdl.^2 .+ SEspiro_mdl.^2);

# 	CIdiff_mdl = SEdiff_mdl.*critical_level; 
# 	df = sort(DataFrame(val = vec(permutedims(diffcoef_mdl[idxCovarmdl, :])),
# 	idx = 1:size(diffcoef_mdl , 2)), :val)

# 	p1 = confidenceplot(
# 		# vec(permutedims(avgcoef_mdl[idxCovarmdl, df.idx])),
# 		vec(permutedims(avgcoef_mdl[idxCovarmdl, :])),
# 		nameModules[:],
# 		vec((CIdiff_mdl[idxCovarmdl, :])),
# 		xlabel = xCovarFig_mdl*" Effect Size", legend = false,
# 		fontfamily = myfont,
# 		title = "Avg", xlim = (-.3,.35) ,
# 		# size = (500, 600),	
# 	)
	
# 	plot(p1)
# 	y2 = collect(1:length(nameModules)).-0.85;
# 	x2 = vec(permutedims(diffcoef_mdl[idxCovarmdl, :]));
# 	ε2 = vec((CIdiff_mdl[idxCovarmdl, :]));
# 	v_significant2 = (x2+ε2).*(x2-ε2) .> 0;
# 	scatter!(
# 	        x2, 
# 	        y2, 
# 	        xerror = ε2,
# 	        legend = false,
# 	        grid = false,
	        
# 	        # yaxis = false,
# 	        y_foreground_color_axis = :white,
# 	        y_foreground_color_border = :white,
# 	        ylims = (-0.5,length(nameModules)+0.10),
	
# 	        markershape = :circle,
# 	        zcolor = v_significant2,
# 	        color =  cgrad([:orange, :red]),
# 	        linecolor = :black,
	
# 	        title = "Avg - Diff",
	
# 	        # size = (400,300),
# 	    )
# end

# ╔═╡ 884e5cdd-cc55-45d6-be04-6ab1671ceaf6
begin
	if (@isdefined TstatZmdl) 
		# namesX = permutedims(vCovarNames[(vCovarNames[1] == "Intercept" ? 2 : 1):end])
		# namesZmdl = string.(permutedims(nameModules))
		# plot(transpose(TstatZmdl[idxCovar,1:end]), # [:, 1:2]
		#     label= namesX, xlabel= "Metabolite Classes by module", ylabel = "T-statistics",
		#     legend =:outerright, lw = 2, marker = :circle, grid = false,
		#     xticks = (collect(1:size(mZmdl)[2]), namesZmdl),xrotation = 45,
		# 	size = (800,550),
		# 	# size = (650,550),
			
		#     foreground_color_legend = nothing, # remove box of legend   
		#     bottom_margin = 10mm, format = :svg,
		#     title = string("T-statistics of Coefficients Estimates") )
		# hline!([2], color= :red, label = "")
		# hline!([-2], color= :red, label = "")
				
		
		# # namesX = permutedims(vCovarNames[(vCovarNames[1] == "Intercept" ? 2 : 1):end])
		# namesZsub = string.(permutedims(levelsSub[1:end]))
		# plot(transpose(TstatZsb[:,1:end])[:,2:end], # [:, 1:2]
		#     label= namesX, xlabel= "Super Pathway", ylabel = "T-statistics",
		#     legend =:outerright, lw = 2, marker = :circle, grid = false,
		#     xticks = (collect(1:size(mZsub)[2]), namesZsub),xrotation = 45,
		# 	size = (800,550),
		#     foreground_color_legend = nothing, # remove box of legend   
		#     bottom_margin = 10mm, format = :svg,
		#     title = string("T-statistics of Coefficients Estimates") )
		# hline!([2], color= :red, label = "")
		# hline!([-2], color= :red, label = "")
	end
end

# ╔═╡ 0e61db2d-a078-46de-98b5-1358dd7d52be
let
if (@isdefined TstatZmdl) && (xCovariates != ["Intercept"])
	catTstatZmdl = zeros(size(TstatZmdl,1), size(TstatZmdl,2));
	catTstatZmdl[findall(abs.(TstatZmdl) .<2)].= 0;
	catTstatZmdl[findall((TstatZmdl .>=2) .&& (TstatZmdl .<3))].= 1;
	catTstatZmdl[findall((TstatZmdl .<=-2) .&& (TstatZmdl .>-3))].= -1;
	catTstatZmdl[findall(TstatZmdl .>=3)].= 2;
	catTstatZmdl[findall(TstatZmdl .<=-3)].= -2;
	catTstatZmdl
	
	# myscheme = [RGB{Float64}(0.792, 0, 0.125),
 #            RGB{Float64}(0.957, 0.647, 0.51),
 #            RGB{Float64}(0.5, 0.5, 0.5),    
 #            RGB{Float64}(0.557, 0.773, 0.871),
 #            RGB{Float64}(0.02, 0.443, 0.69)];
	# labels
	# namesTstsX = vCovarNames
	# namesXtbl = reshape(namesTstsX[2:end], 1, length(namesTstsX)-1)
	# clrbarticks = ["T-stats ≤ -3", "-3 < T-stats ≤ -2", "-2 < T-stats < 2", "2 ≤ T-stats < 3", "3 ≤ T-stats"];
	# n_cat = length(clrbarticks)
	# yt = range(0,1,n_cat+1)[1:n_cat] .+ 0.5/n_cat
	# myfont = "Computer Modern" 

	# l = @layout [ a{0.50w} [b{0.1h}; c{0.2w}] ];

	# mycolors =ColorScheme(myscheme)
	# colors = cgrad(mycolors, 5, categorical = true);

	# PLOT MAIN HEATMAP
	p1 = mlmheatmap(permutedims(catTstatZmdl[2:end,:]), permutedims(TstatZmdl[2:end,:]), 
	    xticks = (collect(1:size(mZmdl)[1]), namesXtbl), xrotation = 25,
	    yticks = (collect(1:size(mZmdl)[2]), nameModules), tickfontfamily = myfont ,
	    color = colors, cbar = false,
		clims = (-3, 3),
	    annotationargs = (10, "Arial", :lightgrey), 
	    linecolor = :white , linewidth = 2);

	# PLOT GAP 
	p2 = plot([NaN], lims=(0,1), framestyle=:none, legend=false);

	# ANNOTATE COLOR BAR TITLE
	annotate!(0.5, -3.02, text("T-statistics", 10, myfont))
	
	# PLOT HEATMAP COLOR BAR
	xx = range(0,1,100)
	zz = zero(xx)' .+ xx
	p3 = Plots.heatmap(xx, xx, zz, ticks=false, ratio=3, legend=false, 
	            fc=colors, lims=(0,1), framestyle=:box, right_margin=15mm);
	
	# ANNOTATE COLOR BAR AXIS
	[annotate!(2.15, yi, text(ti, 7, myfont)) for (yi,ti) in zip(yt,clrbarticks)]
	
	# PLOT MLMHEATMAP
	plot(p1, p2, p3, layout=l, margins=2mm)
end
end

# ╔═╡ 0bea76b5-5fc2-4a33-9143-5b10c026f335
md"""
## References

	Liang, J. W., Nichols, R. J., Sen, Ś.,  Matrix Linear Models for High-Throughput Chemical Genetic Screens, Genetics, Volume 212, Issue 4, 1 August 2019, Pages 1063–1073, https://doi.org/10.1534/genetics.119.302299

	Dieterle F, Ross A, Schlotterbeck G, Senn H. Probabilistic Quotient Normalization as Robust Method to Account for Dilution of Complex Biological Mixtures. Application in 1H NMR Metabonomics. Analytical Chemistry. 2006;78: 4281–4290. doi:10.1021/ac051632c

	Gillenwater, Lucas A., Katerina J. Kechris, Katherine A. Pratte, Nichole Reisdorph, Irina Petrache, Wassim W. Labaki, Wanda O’Neal, Jerry A. Krishnan, Victor E. Ortega, Dawn L. DeMeo, and Russell P. Bowler. 2021. "Metabolomic Profiling Reveals Sex Specific Associations with Chronic Obstructive Pulmonary Disease and Emphysema" Metabolites 11, no. 3: 161. [https://doi.org/10.3390/metabo11030161](https://doi.org/10.3390/metabo11030161)

	Hastie, T., Tibshirani, R., Sherlock, G., Eisen, M., Brown, P. and Botstein, D., Imputing Missing Data for Gene Expression Arrays, Stanford University Statistics Department Technical report (1999), http://www-stat.stanford.edu/~hastie/Papers/missing.pdf Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing value estimation methods for DNA microarrays BIOINFORMATICS Vol. 17 no. 6, 2001 Pages 520-525
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataFramesMeta = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
FreqTables = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MatrixLM = "37290134-6146-11e9-0c71-a5c489be1f53"
Missings = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
MultipleTesting = "f8716d33-7c4a-5097-896f-ce0ecbd3ef6b"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsModels = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.4"
ColorSchemes = "~3.24.0"
DataFrames = "~1.3.4"
DataFramesMeta = "~0.14.1"
Distributions = "~0.25.62"
FileIO = "~1.16.1"
FreqTables = "~0.4.5"
Images = "~0.25.2"
Latexify = "~0.15.15"
MatrixLM = "~0.1.3"
Missings = "~1.1.0"
MultipleTesting = "~0.6.0"
Plots = "~1.30.0"
PlutoUI = "~0.7.39"
PrettyTables = "~1.3.1"
RecipesBase = "~1.3.4"
StatsBase = "~0.33.16"
StatsModels = "~0.6.30"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "cd45f8ca5320bfb2c1bdb7352b84619889a4ade1"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.Chain]]
git-tree-sha1 = "8c4920235f6c561e401dfe569beb8b924adad003"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.5.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "05f9816a77231b07e634ab8715ba50e5249d6f76"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[deps.DataFramesMeta]]
deps = ["Chain", "DataFrames", "MacroTools", "OrderedCollections", "Reexport"]
git-tree-sha1 = "6970958074cd09727b9200685b8631b034c0eb16"
uuid = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
version = "0.14.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9242eec9b7e2e14f9952e8ea1c7e31a50501d587"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.104"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "ec22cbbcd01cba8f41eecd7d44aac1f23ee985e3"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.2"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreqTables]]
deps = ["CategoricalArrays", "Missings", "NamedArrays", "Tables"]
git-tree-sha1 = "4693424929b4ec7ad703d68912a6ad6eff103cfe"
uuid = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
version = "0.4.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "97829cfda0df99ddaeaafb5b370d6cab87b7013e"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d53480c0793b13341c40199190f92c611aa2e93c"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.2"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "424a5a6ce7c5d97cca7bcc4eac551b97294c54af"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.9"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "899050ace26649433ef1af25bc17a815b3db52b7"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.9.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "3447781d4c80dbe6d71d239f7cfb1f8049d4c84f"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.6"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "b0b765ff0b4c3ee20ce6740d843be8dfce48487c"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.3.0"

[[deps.ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "e7c68ab3df4a75511ba33fc5d8d9098007b579a8"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.2"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "44664eea5408828c03e5addb84fa4f916132fc26"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.1"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "5fa9f92e1e2918d9d1243b1131abe623cdf98be7"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.3"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "31d6adb719886d4e32e38197aae466e98881320b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "9bbb5130d3b4fa52846546bca4791ecbdfb52730"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.38"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixLM]]
deps = ["DataFrames", "Distributed", "GLM", "LinearAlgebra", "Random", "SharedArrays", "Statistics", "Test"]
git-tree-sha1 = "742d34bc491fa02269dd019a0fed8f7e40e549c0"
uuid = "37290134-6146-11e9-0c71-a5c489be1f53"
version = "0.1.3"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultipleTesting]]
deps = ["Distributions", "SpecialFunctions", "StatsBase"]
git-tree-sha1 = "1e98f8f732e7035c4333135b75605b74f3462b9b"
uuid = "f8716d33-7c4a-5097-896f-ce0ecbd3ef6b"
version = "0.6.0"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "6d42eca6c3a27dc79172d6d947ead136d88751bb"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "3ef8ff4f011295fd938a521cb605099cecf084ca"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.15"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d0a61518267b44a70427c0b690b5e993a4f5fe01"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.30.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "9a46862d248ea548e340e30e2894118749dc7f51"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.5"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "792d8fd4ad770b6d517a13ebb8dadfcac79405b8"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.6.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "2aded4182a14b19e9b62b063c0ab561809b5af2c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.8.0"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "a5e15f27abd2692ccb61a99e0854dfb7d48017db"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.33"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─d9a9dd61-4cab-4ee0-a769-7dfc45c281b5
# ╟─34d05aeb-63fc-46c5-aefe-b9f6622e6c9b
# ╠═148df6fb-feac-4fe2-9ccc-18cb460796bf
# ╟─a59b3bc2-edaa-11ec-3b2d-fd5bdd0ee1f2
# ╟─58cdac80-d63b-4c3e-9193-d507a0da9be6
# ╟─de5b8917-915c-4c17-8d2c-d55f06822a3f
# ╟─9da0827e-aec0-4981-9139-aee445104a04
# ╟─4e368004-d055-40a1-a28d-c0f4ca55dad7
# ╟─a5ce8064-1af1-47cc-8dcd-db7d96f3d537
# ╟─e176af8a-8bc3-497d-91f8-aa7259ab3c48
# ╟─32b95e81-6761-4b2b-a59b-1eedc9a1443e
# ╟─ae4fd58c-cd3d-4b7d-8c29-dd84e2a103c9
# ╟─96741af6-faa9-40ef-85e2-8647199ed6ab
# ╟─1f367dfd-8642-4c2c-806b-42388725f0cf
# ╟─19b7b087-ce29-43f4-b870-299233045d3a
# ╟─3f703537-ef4f-4d2d-a348-6ac4332abdaa
# ╟─265e748a-f17d-4a52-a6b7-4c0537fccc86
# ╟─8ef72518-4b30-4efe-9415-d84719d39be4
# ╟─f679bf30-78a6-48c0-a17d-391a0e185289
# ╟─3e5e2657-253c-4f96-a5f3-3aea07485478
# ╟─01f9a0b1-4d33-49a5-bf7a-fec0b04ad1a6
# ╟─6902142b-a049-4724-aa1b-ee5b167c0cd6
# ╟─859546bf-7508-45ab-b10e-412314b6d753
# ╟─521b36b1-965a-483e-b67c-b78af8ea129b
# ╟─5e0c72da-44db-4f92-9002-03682ac3843c
# ╟─ade4dd1e-99c8-4948-9922-21241c29561f
# ╟─764dd801-f96c-4d9b-95df-27f6e339bb41
# ╟─4c0a47e9-8dc5-4635-9449-39dfd3b1eee5
# ╟─c5b66499-9a39-4181-b177-6c3b0c563fc9
# ╟─6952ec52-982d-4a37-ae1a-e7f5f1972fda
# ╟─1f9d797d-5832-4e28-bfe5-056cf3e70a33
# ╟─85236909-e107-4aa7-8b19-57892f4610d0
# ╟─5981d0b3-3de7-43cf-af1f-51cfc0179014
# ╟─807aa3cf-0898-4e20-8b19-d5627d4050e1
# ╟─1d314d3f-3b2a-4e6f-9193-a1e0bd456a93
# ╟─5a5ed093-fa57-4666-bcf5-3e30f1a1182f
# ╟─69389cae-cc6e-4be4-8f54-3d7d0f68c0ed
# ╟─9bfc2ec5-a235-43c6-a0af-d7287dfef304
# ╟─27cee0f5-b86f-404a-9dd8-61bdf72bc43f
# ╟─b3278187-f10b-4f16-a8b3-f1bd4e0ad854
# ╟─49d9f10e-e9a2-4541-9418-e67040592dad
# ╟─6b9213b8-d7ac-44c1-b0ae-a5e19e8e3d45
# ╟─af5ef391-b9bf-41e8-b539-5fa58f761765
# ╟─1406d884-c1e2-4e84-b5d2-bb3bd0587e39
# ╟─ce038faa-ab1f-4686-b14d-2c7f8ba0a257
# ╟─b180bc39-33ca-41e6-9240-f69ebdc1085c
# ╟─440f93f3-6c51-4dea-bd69-a6387bdfeecc
# ╟─4be72887-8c6d-4812-80f8-6c75e55d73ee
# ╟─87f8fc10-c6f0-4487-9394-d22e3b480258
# ╟─f5517a21-4885-447c-800f-a065e387e935
# ╟─cb63e350-0f68-457c-8098-1d57000e40c1
# ╟─a0b8461a-493f-4ee2-921b-7cc3c3b30ce4
# ╟─b9a70c70-a560-419a-b86b-df252df9ef85
# ╟─a321e2ce-0307-4baa-9d9e-821236a84f36
# ╠═923b38bb-9f09-4591-a229-6552fd756307
# ╠═660f3468-00ba-4938-83a6-9bdbb20d4795
# ╟─d427bc34-94a4-48bb-b880-d604a67a8d4a
# ╟─0f942bb6-ea42-4988-93f4-6afa3874f91a
# ╟─202b9efe-a41a-4ade-929f-c719a4a17abd
# ╠═7d3574f1-a6ff-4629-8fe0-b40ba4de22a8
# ╟─e09bd194-ccb2-4448-8e05-1816dc261e31
# ╟─f91ab58d-0808-4558-824d-6f348316c1b4
# ╠═e76052ae-be3b-4459-be18-6c4ecc903093
# ╠═7d6c2a28-4f85-4939-a7c7-f5aeab385d7e
# ╠═0c94e2c0-3e65-47a5-9093-822ba610e9f2
# ╟─a65d4195-8580-4a6e-985d-1afa0d407f78
# ╠═c8f666d6-953c-4d53-99aa-6ec72fd82669
# ╠═1ee7e175-1b6d-4f08-8ec7-945394e36311
# ╠═2facc8b7-0300-4b31-bb51-e9c4ee2b7fd6
# ╠═c610e158-1028-47e7-877f-7a6f7e863cb3
# ╠═4cc3ecef-a047-4362-af4b-1e7e78564823
# ╠═4b0bebfc-3162-427f-8430-f86eaeaa0b3c
# ╠═d8885ae3-162d-49ea-b3f5-1030d76f73a0
# ╠═b9bc72e7-e539-4715-9c75-bcd9ada7182b
# ╠═6b89cc56-234f-4246-a200-fed293c7fffc
# ╠═97af3d9f-1ed3-4bed-a46e-1682429c6355
# ╠═da2a35a9-1dbd-4bd5-acbc-23f51ddc250e
# ╠═ddf1313f-ecef-4970-9894-3ef2c832a33a
# ╠═ea420bf4-8e1b-4633-8ac3-75fd7f73ef91
# ╠═38f8472e-5930-4ea8-ae28-73781055c08b
# ╠═29e5234b-0459-4531-9c5b-eff5a7eb8a67
# ╠═60ef862a-f67a-4ea4-ae6d-74a5d2acfe93
# ╠═37c86980-6241-4121-ba87-6a3d8dc5a1ca
# ╠═5d92b5bd-fc46-49d1-b1fc-c13c00433265
# ╟─df1598ed-d2bc-44eb-8c66-fbc57b262d41
# ╟─d161fce0-14dd-44db-ba83-a6f58f4d9bdf
# ╟─c090c950-08cf-4b3f-a06d-e96f51b1b52c
# ╟─c5b22684-0221-4d17-b6e2-0196f0053f87
# ╟─08836fbc-577e-4793-86ab-cdae1bf08f56
# ╟─f572cc42-e003-40a2-be0b-11b52a6cc946
# ╟─77171c73-5281-4c72-92cd-459faedece45
# ╟─287da42a-4294-4942-8f33-7b478cddfbfd
# ╠═44a53807-3ff5-423b-9745-18be61fc2af3
# ╟─afad2cfb-d394-43fb-b434-f4de6bd4dff6
# ╟─f1c0cd37-362b-4fe6-b8e3-37ba3886ce2d
# ╟─884e5cdd-cc55-45d6-be04-6ab1671ceaf6
# ╟─0e61db2d-a078-46de-98b5-1358dd7d52be
# ╟─0bea76b5-5fc2-4a33-9143-5b10c026f335
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
