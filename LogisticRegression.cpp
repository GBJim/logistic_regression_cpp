#include <iostream>
#include <numeric>
#include <assert.h>
#include <math.h>
#include "LogisticRegression.h"

#define X_COLUMN_SIZE 15
#define OUTPUT_SIZE 3

float sigmoid(float x){
    return  1 / (1 + exp(-x));
}

LogisticRegression::LogisticRegression(Eigen::MatrixXf _coef, Eigen::VectorXf _intercept)
    :coef(_coef),
    intercept(_intercept)
{
    std::cout << "Initalization" << std::endl;
    assert(coef.cols()==intercept.rows());
}

Eigen::MatrixXf LogisticRegression::PredictProba(const Eigen::MatrixXf& x)
{      

    assert(x.cols()==coef.rows());
    Eigen::MatrixXf proba = DecisionFunction(x).unaryExpr(&sigmoid);
    for(int i=0; i<proba.rows(); i++)    
    {   
        proba.row(i) /= proba.row(i).sum();
    }
  
    return  proba; 
     
}

Eigen::MatrixXf LogisticRegression::DecisionFunction(const Eigen::MatrixXf& x)
{  
    Eigen::MatrixXf decisionMatrix = x * coef;
    
    for(int i=0; i<decisionMatrix.rows(); i++)    
    {
        decisionMatrix.row(i) += intercept.transpose();
    }

    return decisionMatrix;
}

int main()
{    
    const int COEF_ROW_SIZE = X_COLUMN_SIZE;
    const int COEF_COLUMN_SIZE = 3;
    Eigen::MatrixXf coef(COEF_ROW_SIZE, COEF_COLUMN_SIZE); 
    Eigen::VectorXf intercept(COEF_COLUMN_SIZE); 

    coef  << 5.03119111, -3.37148653, -2.55838091,
        4.74920434, -2.18773315, -3.41168266,
       -0.57888242,  0.89042858, -0.34036991,
        4.57855364, -2.8240788 , -2.58635392,
        4.44865641, -2.35501087, -2.89060287,
       -0.6695677 ,  1.01989   , -0.39092799,
        4.29840305, -2.56192869, -2.55442857,
        4.24002566, -2.51431002, -2.52731369,
       -0.6766406 ,  1.03569387, -0.40170813,
        4.25595993, -2.41199168, -2.67098104,
        4.18971945, -2.72296708, -2.28146283,
       -0.66042445,  1.01651794, -0.38876421,
        4.38806922, -2.21071031, -3.05863649,
        4.19870368, -3.02405522, -2.03187972,
       -0.56237788,  0.8596787 , -0.33071437;
    intercept << -8.84462181,  3.78325803,  4.1552604;

    LogisticRegression logisticRegressor(coef, intercept);
    Eigen::MatrixXf x(2, X_COLUMN_SIZE);
       
    x << -0.08999475814902172,
  0.4719592615271063,
  -0.0008914611129656215,
  0.2533444832119072,
  0.11093649493392026,
  0.00013898185178996136,
  -0.09002301645848712,
  0.471987509296827,
  -0.001477574247538228,
  -0.09003253427028862,
  0.4719970376490806,
  -0.0019033202730159501,
  -0.09003899542065372,
  0.47200348825863986,
  -0.0008480471307191713,
 0.25333770643246917,
  0.11093352746390366,
  0.00013897813412718245,
  -0.09002060840870779,
  0.47197488397652204,
  -0.001477534723508922,
  -0.09003012596591481,
  0.4719844120738992,
  -0.0019032693605921757,
  -0.09003658694344886,
  0.4719908625109094,
  -0.0008480244460793278,
  -0.09004000188094963,
  0.47199427744929434,
  -0.0008625250848484312;
  std::cout << logisticRegressor.PredictProba(x) << std::endl;  
    return 0;
}

