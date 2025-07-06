cmake_minimum_required(VERSION 3.10)
project(TargetTracker LANGUAGES CXX)

# Настройки компилятора
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# Автоматическое обнаружение исходных файлов
file(GLOB_RECURSE HEADER_FILES "include/*.h")

file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

if(NOT SOURCE_FILES)
    message(FATAL_ERROR "No source files found in src/ directory!")
endif()

if(NOT HEADER_FILES)
    message(WARNING "No header files found in include/ directory!")
endif()

# Поиск OpenCV (основная зависимость)
find_package(OpenCV REQUIRED)

# Создание исполняемого файла
add_executable(${PROJECT_NAME} 
    ${SOURCE_FILES}
    ${HEADER_FILES}
    main.cpp
)

# Подключение заголовков
target_include_directories(${PROJECT_NAME} PRIVATE  include/ )

# Линковка всех зависимостей.
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )
