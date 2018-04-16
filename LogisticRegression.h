#include <vector> 
#include <Eigen/Dense>  



class LogisticRegression    
{
    public:
        LogisticRegression(Eigen::MatrixXf _coef, Eigen::VectorXf _intercept);
        Eigen::MatrixXf PredictProba(const Eigen::MatrixXf& x);

    private:
        LogisticRegression(){}
        Eigen::MatrixXf DecisionFunction(const Eigen::MatrixXf& x); 
        Eigen::MatrixXf coef;
        Eigen::MatrixXf intercept;

};


