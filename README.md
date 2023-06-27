# Machine-Learning-Valuation
Utilizing machine learning for company valuation I use various financial indicators to predict the log equity market value, lnME, and form trading strategies based on the model's out-of-sample performance.

Please go through the "questions asked" to understand what all I did in depth and "Solution" to understand what the output tells us.

This problem is focused on applying machine learning techniques for stock valuation and forming trading strategies. It requires the prediction of current log equity market value (lnME) using various characteristics like log book value, issue size, profitability, investment, leverage, momentum, return on equity, and risk value, along with their squared and interaction terms. The problem also explores the use of Elastic Net procedure for regularizing regression, creating mispricing variables, and performing Fama-MacBeth regressions to determine if these signals can predict returns effectively.

Steps I performed:

I first accessed the DT_StockRetAcct.csv dataset on BruinLearn and started to code a machine learning model for valuation using comparables.

I predicted the current log equity market value, lnME, by creating features. I started with creating a log book value, lnBE, by adding the log market value to the log book-to-market ratio (lnBE = lnBM + lnME).

Then, I considered eight different financial characteristics: lnBE, lnIssue, lnProf, lnInv, lnLever, lnMOM, lnROE, and rv. For each of these features, except lnBE, I created additional characteristics as the squared value of the original characteristic and named them with a "2" at the end (e.g., lnProf2).

To these features, I added interaction terms where I multiplied each characteristic with lnBE (except lnBE itself to avoid redundancy).

Additionally, I created 11 dummy variables using the ff_ind industry variable, skipping the 12th industry as my regressions would run with intercepts. This resulted in a total of 34 features to predict lnME.

For each year in the sample, I ran a cross-sectional regression of lnME on these features. I recorded the predicted values lnME_hat from this regression each year and plotted the R2 from these regressions across the years, tracking the model's ability to explain equity market values across firms.

I created a variable z_OLS as a measure of mispricing, calculating it as the actual market value minus the predicted market value for each firm each year (z_OLS = lnME – lnME_hat).

Using the Elastic Net procedure (with alpha or l1_ratio set at 0.5), I ran cross-validation exercises with 10 folds for each year, estimating lnME-hat. I identified the optimal regularization parameter and ran the Elastic Net procedure for all firms for that year. I plotted the chosen regularization parameters for each year.

I collected the predicted market values from the Elastic Net procedure (lnME_hat_EN) and then created another mispricing variable z_EN for each firm and year (z_EN = lnME – lnME_hat_EN).

I created firm excess returns as ExRet = lnAnnRet – lnRf, which represent next year’s return. Each year, I ran a cross-sectional regression of ExRet on an intercept and the mispricing variables z_OLS and z_EN.

Next, I ran the Fama-MacBeth regression to get the portfolio returns based on sorts on these variables, reporting the slope in the Fama-MacBeth regression, as well as their t-statistics.

I evaluated these signals, z_OLS or z_EN, to assess their utility in predicting returns and identified the one that seemed best.

Finally, based on which gave the highest portfolio Sharpe ratio (either z_OLS or z_EN), I ran the Fama-MacBeth regressions including lnBM, lnProf, lnInv, lnMom, as well as industry dummies on the right hand side, to evaluate if the chosen mispricing signal was still useful.
