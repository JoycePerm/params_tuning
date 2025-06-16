// deploy/inference.cpp
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/optim/sgd.h>
#include <torch/nn/modules/loss.h>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

std::map<std::string, std::vector<std::string>> DATA_KEY = {
    {"x", {"WorkSpeed", "RealPower", "Focus", "CutHeight", "GasPressure"}},
    {"y", {"hem", "flySlag", "bubbleSlag", "stratification", "roughness", "breach", "fluidSlag", "slag", "scum", "burrs", "overBreach"}},
    {"cd", {"GasType", "NozzleName"}},
    {"cc", {"LaserPower", "Thickness", "NozzleDiameter"}}};

struct HyperParams
{
    torch::Tensor x_mean;
    torch::Tensor x_std;
    torch::Tensor cc_mean;
    torch::Tensor cc_std;
};

struct ConfigData
{
    int wf;
    std::vector<float> X;
    std::vector<float> CC;
    std::vector<int> CD; // 假设cd包含字符串类型数据
};

// std::pair<std::string, std::string> get_paths(int wf)
// {
//     const std::string base_dir = "/home/chenjiayi/code/params-tuning";
//     if (wf == 0)
//     {
//         return {
//             base_dir + "/log/type0_LN/last_model.pt",
//             base_dir + "/data/type0/hyper-params.json"};
//     }
//     else
//     {
//         return {
//             base_dir + "/log/type1_LN/last_model.pt",
//             base_dir + "/data/type1/hyper-params.json"};
//     }
// }

HyperParams load_hyper_params(const std::string &json_path)
{
    std::ifstream file(json_path);
    if (!file.is_open())
    {
        throw std::runtime_error("无法打开JSON文件: " + json_path);
    }

    json data;
    file >> data;

    // 提取x参数
    std::vector<float> x_means, x_stds;
    for (const auto &key : DATA_KEY.at("x"))
    {
        x_means.push_back(data[key]["mean"]);
        x_stds.push_back(data[key]["std"]);
    }

    // 提取cc参数
    std::vector<float> cc_means, cc_stds;
    for (const auto &key : DATA_KEY.at("cc"))
    {
        cc_means.push_back(data[key]["mean"]);
        cc_stds.push_back(data[key]["std"]);
    }

    return {
        torch::tensor(x_means),
        torch::tensor(x_stds),
        torch::tensor(cc_means),
        torch::tensor(cc_stds)};
}

ConfigData load_config(const std::string &yaml_path)
{
    ConfigData config;

    try
    {
        // 加载YAML文件
        YAML::Node yaml_config = YAML::LoadFile(yaml_path);

        // 读取WF值
        config.wf = yaml_config["WF"].as<int>();

        // 读取X参数
        for (const auto &key : DATA_KEY.at("x"))
        {
            config.X.push_back(yaml_config[key].as<float>());
        }

        // 读取CC参数
        for (const auto &key : DATA_KEY.at("cc"))
        {
            config.CC.push_back(yaml_config[key].as<float>());
        }

        // 读取CD参数
        for (const auto &key : DATA_KEY.at("cd"))
        {
            config.CD.push_back(yaml_config[key].as<int>());
        }
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "YAML解析错误: " << e.what() << std::endl;
        throw;
    }

    return config;
}

auto optimize_input(torch::Tensor x, torch::Tensor c, torch::Tensor y, std::string pt_path, torch::Tensor st, float lr = 0.01, int steps = 1)
{

    torch::jit::Module model;
    try
    {
        model = torch::jit::load(pt_path);
        // model = torch::jit::optimize_for_inference(model);
        model.eval();
    }
    catch (const std::exception &e)
    {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        // return -1;
    }

    torch::Tensor t_x = x.clone().set_requires_grad(true);

    std::vector<torch::Tensor> params = {t_x};
    auto optimizer = torch::optim::SGD(params, torch::optim::SGDOptions(lr));

    auto loss = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kSum));

    std::vector<torch::jit::IValue> inputs = {t_x, c};

    torch::Tensor y_pred;
    torch::Tensor dx;

    for (int i = 0; i < steps; ++i)
    {
        y_pred = model.forward(inputs).toTensor();

        torch::Tensor test_loss = loss(y_pred, y).mean();

        optimizer.zero_grad();
        test_loss.backward();
        optimizer.step();

        dx = t_x - x;
        dx = dx * st;
    }
    torch::Tensor output = y_pred.clone().detach() * 5;

    std::cout << "输出结果:\n"
              << std::fixed << std::setprecision(4) << output << std::endl;

    torch::Tensor f_dx = dx.detach();

    return f_dx;
}

int main()
{
    ConfigData config = load_config("../../config.yaml");
    torch::Tensor X_tensor = torch::tensor(config.X);
    torch::Tensor CC_tensor = torch::tensor(config.CC);
    torch::Tensor CD1_tensor = torch::tensor(config.CD[0]);
    torch::Tensor CD2_tensor = torch::tensor(config.CD[1]);

    // std::cout << "WF: " << config.wf << std::endl;
    // std::cout << "X: " << X_tensor << std::endl;
    // std::cout << "CC: " << CC_tensor << std::endl;
    // std::cout << "CD: " << config.CD << std::endl;

    std::string pt_path;
    std::string hyperp_path;

    // 模型路径
    if (config.wf == 0)
    {
        pt_path = "../contnet_fused_0.pt";
        hyperp_path = "/home/chenjiayi/code/params-tuning/data/type0/hyper-params.json";
    }
    else
    {
        pt_path = "../contnet_fused_1.pt";
        hyperp_path = "/home/chenjiayi/code/params-tuning/data/type1/hyper-params.json";
    }

    HyperParams params = load_hyper_params(hyperp_path);

    // std::cout << "x_mean:\n"
    //           << params.x_mean << "\n"
    //           << "x_std:\n"
    //           << params.x_std << "\n"
    //           << "cc_mean:\n"
    //           << params.cc_mean << "\n"
    //           << "cc_std:\n"
    //           << params.cc_std << std::endl;

    torch::Tensor x = ((X_tensor - params.x_mean) / params.x_std).to(torch::kFloat); // batch_size=1, feature_dim=5
    x = x.unsqueeze(0);
    torch::Tensor cc = ((CC_tensor - params.cc_mean) / params.cc_std).to(torch::kFloat);
    cc = cc.unsqueeze(0);
    torch::Tensor cd_0 = torch::nn::functional::one_hot(CD1_tensor, 4);
    cd_0 = cd_0.unsqueeze(0);
    torch::Tensor cd_1 = torch::nn::functional::one_hot(CD2_tensor, 3);
    cd_1 = cd_1.unsqueeze(0);

    // std::cout << "cc: " << cc.sizes() << std::endl;
    // std::cout << "cd_0: " << cd_0.sizes() << std::endl;
    // std::cout << "cd_1: " << cd_1.sizes() << std::endl;

    torch::Tensor c = torch::cat({cc, cd_0, cd_1}, 1);

    torch::Tensor y = (torch::tensor({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}) / 5.0).unsqueeze(0);

    torch::Tensor st = torch::tensor({100, 100, 100, 100, 100});

    auto start = std::chrono::high_resolution_clock::now();

    torch::Tensor dx = optimize_input(x, c, y, pt_path, st);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double elapsed_seconds = duration.count();

    std::cout << "推理耗时: " << elapsed_seconds << " 秒" << std::endl;

    dx = dx * params.x_std;
    dx[0][1] = dx[0][1] * 100 / config.CC[0];

    std::cout << "调整参数: 切割速度 激光功率占比 焦点 喷嘴高度 气压\n"
              << "参数调整量：" << dx << std::endl;

    return 0;
}