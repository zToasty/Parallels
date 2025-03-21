#include <iostream>
#include <fstream>
#include <cmath>
#include <functional>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <vector>
#include <random>
#include <future>

// Шаблонные функции для математических операций
template <typename T>
T fun_sin(T arg) { return std::sin(arg); }

template <typename T>
T fun_sqrt(T arg) { return std::sqrt(arg); }

template <typename T>
T fun_pow(T x, T y) { return std::pow(x, y); }

// Шаблонный класс сервера задач с использованием std::future
template <typename T>
class Server {
public:
    using Task = std::packaged_task<T()>;

    Server() : running(false) {}

    void start() {
        running = true;
        worker = std::thread(&Server::process_tasks, this);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            running = false;
        }
        cv.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }

    std::future<T> add_task(Task task) {
        auto future = task.get_future();
        {
            std::lock_guard<std::mutex> lock(mtx);
            task_queue.push(std::move(task));
        }
        cv.notify_one();
        return future;
    }

private:
    std::queue<Task> task_queue;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    bool running;

    void process_tasks() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return !task_queue.empty() || !running; });

                if (!running && task_queue.empty()) return;

                task = std::move(task_queue.front());
                task_queue.pop();
            }
            task();  // Выполняем задачу, результат автоматически передается в future
        }
    }
};

class Client {
public:
    Client(Server<double>& srv, int id, int num_tasks, std::function<double(double)> task_func, const std::string& filename)
        : server(srv), client_id(id), num_tasks(num_tasks), task_func(task_func), filename(filename), is_two_args(false) {}

    Client(Server<double>& srv, int id, int num_tasks, std::function<double(double, double)> task_func, const std::string& filename, bool is_two_args)
        : server(srv), client_id(id), num_tasks(num_tasks), task_func2(task_func), filename(filename), is_two_args(is_two_args) {}

    void operator()() {
        std::ofstream file(filename);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(1.0, 10.0);

        if (is_two_args) {
            std::vector<std::pair<std::future<double>, std::pair<double, double>>> futures;
            for (int i = 0; i < num_tasks; ++i) {
                double value1 = dist(gen);
                double value2 = dist(gen);
                auto future = server.add_task(std::packaged_task<double()>([=] { return task_func2(value1, value2); }));
                futures.push_back({std::move(future), {value1, value2}});
            }

            for (auto& task : futures) {
                file << "Arg1: " << task.second.first << ", Arg2: " << task.second.second << ", Result: " << task.first.get() << "\n";
            }
        } else {
            std::vector<std::pair<std::future<double>, double>> futures;
            for (int i = 0; i < num_tasks; ++i) {
                double value = dist(gen);
                auto future = server.add_task(std::packaged_task<double()>([=] { return task_func(value); }));
                futures.push_back({std::move(future), value});
            }

            for (auto& task : futures) {
                file << "Arg: " << task.second << ", Result: " << task.first.get() << "\n";
            }
        }

        file.close();
    }

private:
    Server<double>& server;
    int client_id;
    int num_tasks;
    std::function<double(double)> task_func;
    std::function<double(double, double)> task_func2;
    std::string filename;
    bool is_two_args;
};

int main() {
    Server<double> server;
    server.start();

    int N = 100;  // Количество задач на клиента

    std::thread client1(Client(server, 1, N, fun_sin<double>, "sin_results.txt"));
    std::thread client2(Client(server, 2, N, fun_sqrt<double>, "sqrt_results.txt"));
    std::thread client3(Client(server, 3, N, [](double x, double y) { return fun_pow(x, y); }, "pow_results.txt", true));

    client1.join();
    client2.join();
    client3.join();

    server.stop();
    return 0;
}
