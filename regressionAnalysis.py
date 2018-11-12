import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression



class analysisData(object): 

    def __init__(self, filename):
       self.variables = []
       self.filename = filename
       
    def parseFile(self):
        self.dataset = pd.read_csv(self.filename)
        self.variables = self.dataset.columns 
        
dataParser = analysisData('./candy-data.csv') 
dataParser.parseFile()

class LinearAnalysis(object): 

    def __init__(self, targetY): 
    	self.bestX = None
    	self.targetY = targetY
    	self.fit = None

    def runSimpleAnalysis(self, dataParser): 
       
        dataset = dataParser.dataset

        best_pred = 0
        for column in dataParser.variables: 
            if column == self.targetY or column == 'competitorname':
                continue

            x_values = dataset[column].values.reshape(-1,1) 
            y_values = dataset[self.targetY].values

            regr = LinearRegression()
            regr.fit(x_values, y_values)
            preds = regr.predict(x_values) 
            score = r2_score(y_values, preds)

            if score > best_pred:
                best_pred = score
                self.bestX = column
        
        self.fit = best_pred
        print(self.bestX)
        print(self.fit)
        print(regr.coef_)
        print(regr.intercept_)

linear_analysis = LinearAnalysis(targetY='sugarpercent')
linear_analysis.runSimpleAnalysis(dataParser)

class LogisticsAnalysis(object):

	def __init__(self, targetY):
		self.bestX = None
		self.targetY = targetY
		self.fit = None
	
	
	def runsimpleAnalysis(self, dataParser):
		
		dataset = dataParser.dataset
		
		best_pred = 0
		for column in dataParser.variables:
			if column == self.targetY or column == 'competitorname':
				continue
				
			x_values = dataset[column].values.reshape(-1,1)
			y_values = dataset[self.targetY].values
			
			regr = LogisticRegression()
			regr.fit(x_values, y_values)
			preds = regr.predict(x_values)
			score = r2_score(y_values, preds)
			
			if score > best_pred:
				best_pred = score
				self.bestX = column
		
		self.fit = best_pred
		print(self.bestX)
		print(self.fit)
		print(regr.coef_)
		print(regr.intercept_)
        		
        		
    
	def runMultipleRegression(self, dataParser):
         
		dataset = dataParser.dataset
		clean_dataset = dataset.drop([self.targetY, 'competitorname'], axis=1)
		x_values = clean_dataset.values
		y_values = dataset[self.targetY].values

		regr = LogisticRegression()
		regr.fit(x_values, y_values)
		preds = regr.predict(x_values)
		score = r2_score(y_values, preds)

		print(clean_dataset.columns)
		print(score)  
		print(regr.coef_)
		print(regr.intercept_)  
        

            
logistics_analysis = LogisticsAnalysis(targetY='chocolate')
logistics_analysis.runsimpleAnalysis(dataParser)

multivariable_logistics = LogisticsAnalysis(targetY='chocolate')
multivariable_logistics.runMultipleRegression(dataParser)
			
#Problem 3:

# Linear Regression: y = .00440378 + .257063291665

# Logistic Regression: 1/1 + e^ -(0.05901723x - -3.08798586)

# Multiple Regression: -2.52858047x1 -0.19697876x2  0.03940308x3 -0.16539952x4  0.49783674x5 -0.47591613x6
#  0.81511886x7 -0.59971553x8 -0.2581028x9   0.3224988x10   0.05387906x11 -1.68260553x12

# Problem 4:

#(a) What candies contain more sugar, those with caramel or those with chocolate?

# - Dependant = sugar (continuous)

# - Independent = caramel (categorical)

# - Independent = caramel (categorical)

# - Null Hypothesis = caramel and chocolate having same sugar content

#(b) Are there more split-ticket voters in blue states or red states? 

# - Dependant = split ticket voters (discrete)

# - Independent = blue states (discrete)

# - Independent - red states (discrete)

# - Null Hypothesis = equal amount in red/blue states

#(c) Do phones with longer battery life sell at a higher or lower rate than other phones?

# - Dependant = rate (discrete) this could maybe be continuous, but I've never seen a phone sell at $100.12365645646....

# - Independent = other phones (categorical)

# - Independent = phone with longer battery life (categorical)

# - Null Hypothesis = selling at the same rate
			
			
			
			
			
			
			
			
	
	
	
	
	
	
	
	
	
	