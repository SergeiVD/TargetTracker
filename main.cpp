#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <target_tracker.h>


int main(int argc, char *argv[])
{
	TargetTracker tracker;
    tracker.setMaxAllowedDepthError(0.8f); // выбросы > 0.8 м игнорируются

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<float> depth_noise(0.0f, 0.2f);
    std::uniform_int_distribution<int> spike_chance(0, 9); // 10% выбросы

    double time = 0.0;
    double dt = 0.1; // 10 Гц

    for (int i = 0; i < 100; ++i)
    {
        // Синусоидальное движение
        float x = std::sin(0.1f * i);
        float y = std::cos(0.1f * i);
        float z = 3.0f + depth_noise(rng); // нормальная глубина с шумом

        // Иногда выбросим глубину
        if (spike_chance(rng) == 0)
        {
            z += (rng() % 2 == 0 ? 5.0f : -4.0f); // выброс
            std::cout << "[!] ВЫБРОС глубины на кадре " << i << ": z = " << z << std::endl;
        }

        tracker.update(cv::Point2f(x, y), z, time);

        cv::Point2f filtered_pos = tracker.getFilteredPosition();
        float filtered_z = tracker.getFilteredDepth();

        std::cout << "Frame " << i
                  << " | Input: (" << x << ", " << y << ", " << z << ")"
                  << " | Filtered: (" << filtered_pos.x << ", " << filtered_pos.y << ", " << filtered_z << ")\n";

        time += dt;
    }

	std::cout<<"hellot"<<std::endl;
    return 0;
}
