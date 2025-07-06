#include "../include/target_tracker.h"

TargetTracker::TargetTracker() : is_initialized_(false), last_timestamp_(0), process_noise_scale_(1.0f), measurement_noise_scale_(1.0f), max_allowed_depth_error_(1.0f)
{
    // 6 состояний: [x, y, vx, vy, z, vz]
    // 3 измерения: [x, y, z]
    kf_.init(6, 3, 0);
    
    // Инициализация матриц
    initializeMatrices();
    
    // Начальные значения для адаптивного фильтра
    position_variance_ = 10.0f;
    depth_variance_ = 1.0f;
    velocity_variance_ = 5.0f;
}

void TargetTracker::initializeMatrices()
{
    // Матрица перехода (const velocity model)
    kf_.transitionMatrix = (cv::Mat_<float>(6, 6) <<
        1, 0, 1, 0, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1);

    // Матрица измерений (измеряем только позицию и глубину)
    kf_.measurementMatrix = (cv::Mat_<float>(3, 6) <<
        1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0);

    // Инициализация ковариационных матриц
    updateNoiseCovariances();
    
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(0.1));
}

void TargetTracker::updateNoiseCovariances()
{
    // 1. Обновление шума измерений (R матрица)
    kf_.measurementNoiseCov = (cv::Mat_<float>(3, 3) <<
        position_variance_, 0, 0,
        0, position_variance_, 0,
        0, 0, depth_variance_);

    // 2. Обновление шума процесса (Q матрица)
    kf_.processNoiseCov = (cv::Mat_<float>(6, 6) <<
        position_variance_, 0, 0, 0, 0, 0,
        0, position_variance_, 0, 0, 0, 0,
        0, 0, velocity_variance_, 0, 0, 0,
        0, 0, 0, velocity_variance_, 0, 0,
        0, 0, 0, 0, depth_variance_, 0,
        0, 0, 0, 0, 0, velocity_variance_);

    // 3. Обновление ковариации ошибки (P матрица)
    cv::Mat errorCov = cv::Mat::zeros(6, 6, CV_32F);
    errorCov.at<float>(0,0) = errorCov.at<float>(1,1) = position_variance_;
    errorCov.at<float>(2,2) = errorCov.at<float>(3,3) = velocity_variance_;
    errorCov.at<float>(4,4) = depth_variance_;
    errorCov.at<float>(5,5) = velocity_variance_;
    kf_.errorCovPost = errorCov;
}

void TargetTracker::update(const cv::Point2f& position, float depth, double timestamp)
{
    if (!is_initialized_)
    {
        initializeTarget(position, depth, timestamp);
        return;
    }
    
    float dt = timestamp - last_timestamp_;
    last_timestamp_ = timestamp;
    
    // Адаптация параметров фильтра
    adaptFilterParameters(position, depth, dt);
    
    // Обновляем матрицу перехода
    updateTransitionMatrix(dt);
    
    // Предсказание и коррекция
    performPredictionAndCorrection(position, depth);
}

void TargetTracker::initializeTarget(const cv::Point2f& position, float depth, double timestamp)
{
    kf_.statePost.at<float>(0) = position.x;
    kf_.statePost.at<float>(1) = position.y;
    kf_.statePost.at<float>(4) = depth;
    is_initialized_ = true;
    last_timestamp_ = timestamp;
    
    // Инициализация скоростей нулями
    kf_.statePost.at<float>(2) = 0;
    kf_.statePost.at<float>(3) = 0;
    kf_.statePost.at<float>(5) = 0;
}

void TargetTracker::adaptFilterParameters(const cv::Point2f& position, float depth, float dt)
{
    // Простейшая адаптация шумов на основе невязки
    cv::Point2f predicted_pos = getFilteredPosition();
    float predicted_depth = getFilteredDepth();
    
    float position_error = cv::norm(position - predicted_pos);
    float depth_error = std::abs(depth - predicted_depth);
    
    // Адаптация шумов измерений
    measurement_noise_scale_ = 1.0f + 0.1f * position_error;
	
	if(depth_error < max_allowed_depth_error_)
	{
		depth_variance_ = 0.1f + 0.5f * depth_error;		
	}
    
    // Адаптация шума процесса (чем больше dt, тем больше шум)
    process_noise_scale_ = 1.0f + dt;
    
    updateNoiseCovariances();
}

void TargetTracker::updateTransitionMatrix(float dt)
{
    kf_.transitionMatrix.at<float>(0,2) = dt;
    kf_.transitionMatrix.at<float>(1,3) = dt;
    kf_.transitionMatrix.at<float>(4,5) = dt;
}

void TargetTracker::performPredictionAndCorrection(const cv::Point2f& position, float depth)
{
    cv::Mat prediction = kf_.predict();

    float predicted_depth = prediction.at<float>(4);
    bool depth_valid = std::abs(depth - predicted_depth) < max_allowed_depth_error_;

    cv::Mat measurement;
    if (depth_valid)
    {
        measurement = (cv::Mat_<float>(3,1) << position.x, position.y, depth);
        kf_.correct(measurement);
    }
    else
    {
        // "замораживаем" глубину на предсказанном значении
        measurement = (cv::Mat_<float>(3,1) << position.x, position.y, predicted_depth);

        // временно увеличим дисперсию глубины
        float saved_depth_variance = depth_variance_;
        depth_variance_ *= 1e3f;
        updateNoiseCovariances();

        kf_.correct(measurement);

        // вернём назад
        depth_variance_ = saved_depth_variance;
        updateNoiseCovariances();
    }
}

// Методы доступа 
float TargetTracker::getFilteredDepth() const
{
	return is_initialized_ ? kf_.statePost.at<float>(4) : 0.0f;
}
    
cv::Point2f TargetTracker::getFilteredPosition() const
{
	return is_initialized_ ? cv::Point2f(kf_.statePost.at<float>(0), kf_.statePost.at<float>(1)) : cv::Point2f(0,0);
}

double TargetTracker::getLastUpdateTime() const
{
	return last_timestamp_;
}

float TargetTracker::getPositionVariance() const
{
	return position_variance_;
}

float TargetTracker::getDepthVariance() const
{
	return depth_variance_;
}

float TargetTracker::getVelocityVariance() const
{
	return velocity_variance_;
}

// Новые методы для настройки
void TargetTracker::setProcessNoiseScale(float scale)
{ 
    process_noise_scale_ = scale; 
    updateNoiseCovariances();
}

void TargetTracker::setMeasurementNoiseScale(float scale)
{ 
    measurement_noise_scale_ = scale; 
    updateNoiseCovariances();
}

void TargetTracker::setPositionVariance(float variance)
{
	position_variance_ = variance;
	updateNoiseCovariances();
}
    
void TargetTracker::setDepthVariance(float variance)
{
	depth_variance_ = variance;
	updateNoiseCovariances();
}
    
void TargetTracker::setVelocityVariance(float variance)
{
	velocity_variance_ = variance;
	updateNoiseCovariances();
}

void TargetTracker::setMaxAllowedDepthError(float error)
{
    max_allowed_depth_error_ = error;
}
