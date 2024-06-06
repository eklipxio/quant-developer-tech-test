# Eklipx.io Quantitative Engineer Technical Test



## Tech Stack & Ethos
The Eklipx frontend is an Angular SPA  and our backend is largely comprised of serverless azure functions, the pricing engine in particular is written in python leveraging the Quantlib library for option pricing. 
As an engineering team we strive to follow TDD where applicable, make heavy use of continuous integration and deployment, our infrastructure deployment is handled by pulumi deploying into azure and we use git for version control. 
 
For the sake of this exercise please use Quantlib for pricing however you are free to use any library you like for plotting, testing and formatting. 
The test comprises of three sections and delivery is via a **pull request** against this repository. 

### Volatility 
The repository contains two volatility surfaces named clean.csv and noisy.csv 
1) Create a volatility surface from the clean.csv file suitable for use with the pricing exercises below.
2) Parse the noisy.csv file and based on your knowledge clean and interpolate the curve so it is suitable for pricing (propose and implement an interpolation method).
3) Plot the interpolated curve and comment on any pertinent transformations where you see fit.

### Options
Using the volatility curve from the clean.csv and the provided usd yield curve rates (usd_yield.csv) please price the following options.

1) Barrier Down & Out
2) Binary Up & In
3) Flexstrip European (also called daily option)

as per the reference data below:

|option type|direction|spot|strike|ko|ki|payoff|
| -- | -- | -- | -- | -- | -- | -- |
|Barrier Down & Out|put|1174.75|1200|1100|-|-|
|Binary Up & In|call|1174.75|1200|-|1350|20|
|Flexstrip European|call|1174.75|1200|-|-|-|

### Simulation
For each of the 3 options you have priced above please simulate the delta, gamma, vega and theta across the curve with particular attention around the barrier. 

*Once completed please submit a pull request and we will aim to review and reach out within 3 days of the submission.* 

> **IMPORTANT NOTE** : Although we encourage using AI in our day to day activity, please refrain from using any AI coding assistant for this exercise. 
