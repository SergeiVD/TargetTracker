#ifndef TARGET_TRACKER_H
#define TARGET_TRACKER_H

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

class TargetTracker
{
public:
    // Конструктор
    TargetTracker();
    
    // Основной метод обновления фильтра
    void update(const cv::Point2f& position, float depth, double timestamp);
    
    // Методы получения отфильтрованных значений
    float getFilteredDepth() const;
    cv::Point2f getFilteredPosition() const;
    double getLastUpdateTime() const;

	float getPositionVariance() const;
    float getDepthVariance() const;
    float getVelocityVariance() const;
    
    // Методы настройки параметров фильтра
    void setProcessNoiseScale(float scale);
    void setMeasurementNoiseScale(float scale);
    void setPositionVariance(float variance);
    void setDepthVariance(float variance);
    void setVelocityVariance(float variance);

	void setMaxAllowedDepthError(float error);

private:
    // Инициализация матриц фильтра
    void initializeMatrices();
    
    // Обновление ковариационных матриц шумов
    void updateNoiseCovariances();
    
    // Инициализация цели при первом обновлении
    void initializeTarget(const cv::Point2f& position, float depth, double timestamp);
    
    // Адаптация параметров фильтра
    void adaptFilterParameters(const cv::Point2f& position, float depth, float dt);
    
    // Обновление матрицы перехода
    void updateTransitionMatrix(float dt);
    
    // Выполнение предсказания и коррекции
    void performPredictionAndCorrection(const cv::Point2f& position, float depth);

    // Фильтр Калмана
    cv::KalmanFilter kf_;
    
    // Состояние инициализации
    bool is_initialized_;
    
    // Временные метки
    double last_timestamp_;
    
    // Параметры адаптации
    float process_noise_scale_;
    float measurement_noise_scale_;
    float position_variance_;
    float depth_variance_;
    float velocity_variance_;

	float max_allowed_depth_error_;
};

#endif // TARGET_TRACKER_H
