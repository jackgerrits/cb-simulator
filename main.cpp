#include <iostream>

#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

#include "vw.h"
#include "parse_example_json.h"

void print_usage(char *exe_name)
{
    std::cerr << "Usage: " << exe_name << " ml_args num_actions num_contexts min_p max_p no_click_cost click_cost p_strategy tot_iter mod_iter rnd_seed\n";
}

struct simulator_example
{
    std::vector<float> pdf;
    int num_actions;
    int shared_context;;

    std::string generate_action_json(int i, int num_actions, int shared_context)
    {
        std::stringstream ss;
        ss << R"({"A":{"Constant":1,"Id":")" << i << R"("},"B":{"Id":")" << i << R"("},"X":{"Constant":1,"Id":")" << num_actions * shared_context + i << "\"}}";
        return ss.str();
    }

    std::string generate_json(int num_actions, int shared_context, std::vector<float> pdf)
    {
        std::stringstream ss;
        ss << R"({"Version": "1","EventId": "1","a":[)";
        for (int i = 1; i <= num_actions; i++)
        {
            ss << i << ",";
        }
        ss.seekp(-1, std::ios_base::end);
        ss << R"(],"c": {"U": {"C": ")" << shared_context << R"("},"_multi":[)";
        for (int i = 1; i <= num_actions; i++)
        {
            ss << generate_action_json(i, num_actions, shared_context);
            ss << ",";
        }
        ss.seekp(-1, std::ios_base::end);

        ss << R"(]},"p":[)";
        for (int i = 0; i <= num_actions; i++)
        {
            ss << pdf[i] << ",";
        }
        ss.seekp(-1, std::ios_base::end);
        ss << R"(]})";

        return ss.str();
    }

    simulator_example(int num_actions, int shared_context, float min_p, float max_p)
        : num_actions(num_actions), shared_context(shared_context), pdf(num_actions, min_p)
    {
        pdf[shared_context] = max_p;
    }

    multi_ex to_multi_ex(vw &all)
    {
        v_array<example *> examples = v_init<example *>();
        examples.push_back(&VW::get_unused_example(&all));

        auto example_str = generate_json(num_actions, shared_context, pdf);

        DecisionServiceInteraction header;
        VW::read_line_decision_service_json<false>(all, examples, (char *)example_str.c_str(), example_str.length(), false, (VW::example_factory_t)VW::get_unused_example, (void *)&all, &header);

        VW::setup_examples(all, examples);

        // multi_ex ret{examples.begin(), examples.end()};
        // examples.delete_v();
        return examples;
    }
};

void run(string ml_args, int tot_iter, int mod_iter, int rnd_seed = 0, int num_contexts = 10, int num_actions = 10, float min_p = 0.03f, float max_p = 0.04f, float no_click_cost = 0.0f, float click_cost = -1.0f, int p_strategy = 0)
{
    std::default_random_engine rd{rnd_seed};
    std::mt19937 eng(rd());
    std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);
    uint64_t merand_seed = 0;

    std::vector<simulator_example> examples;
    for (int i = 0; i < num_contexts; i++)
    {
        examples.emplace_back(num_actions, i, min_p, max_p);
    }

    auto scorer_pdf = std::vector<float>(num_actions);

    int clicks = 0;
    int good_actions = 0;
    int good_actions_since_last = 0;
    float cost;
    auto all = VW::initialize(ml_args + " --quiet");

    for (auto i = 1; i <= tot_iter; i++)
    {
        std::uniform_int_distribution<> context_distribution(0, examples.size() - 1);
        auto context_index = context_distribution(eng);
        auto &sim_example = examples[context_index];
        auto &cost_pdf = sim_example.pdf;
        auto vw_ex = sim_example.to_multi_ex(*all);
        all->predict(vw_ex);
        auto &scores = vw_ex[0]->pred.a_s;
        auto total = 0.0f;
        for (auto &score : scores)
        {
            total += score.score;
            scorer_pdf[score.action] = score.score;
        }

        auto draw = click_distribution(eng) * total;
        auto sum = 0.0;
        uint32_t top_action = 0;

        for (auto &score : scores)
        {
            sum += score.score;
            if (sum > draw)
            {
                top_action = score.action;
                break;
            }
        }

        if (top_action == context_index)
        {
            good_actions += 1;
            good_actions_since_last += 1;
        }

        if (click_distribution(eng) < cost_pdf[top_action])
        {
            cost = click_cost;
            clicks += 1;
        }
        else
            cost = no_click_cost;

        float p_reported = scorer_pdf[top_action];
        switch (p_strategy)
        {
        case 1:
            p_reported = 1.0f / num_actions;
            break;
        case 2:
            p_reported = (std::max)(p_reported, 0.5f);
            break;
        case 6:
            p_reported = (std::max)(p_reported, 0.9f);
            break;
        case 7:
            p_reported = 0.9f;
            break;
        case 13:
            p_reported = 0.5f;
            break;
        case 14:
            p_reported = (std::max)(p_reported, 0.1f);
            break;
        }

        vw_ex[top_action]->l.cb.costs = v_init<CB::cb_class>();
        vw_ex[top_action]->l.cb.costs.push_back({cost, top_action, p_reported, 0.f});

        all->learn(vw_ex);
        all->finish_example(vw_ex);

        if (i % mod_iter == 0 || i == tot_iter)
        {
            std::cout << ml_args << "," << num_actions << "," << num_contexts
                      << "," << no_click_cost << "," << click_cost << "," << p_strategy
                      << "," << rnd_seed << "," << i << "," << clicks / (float)i << ","
                      << good_actions << "," << good_actions_since_last << std::endl;
            good_actions_since_last = 0;
        }
    }
    VW::finish(*all);
}

int main(int argc, char *argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.size() != 11)
    {
        print_usage(argv[0]);
        return 1;
    }

    auto ml_args = args[0];
    int num_actions, num_contexts, p_strategy, tot_iter, mod_iter, rnd_seed;
    float min_p, max_p, no_click_cost, click_cost;

    num_actions = std::stoi(args[1]);
    num_contexts = std::stoi(args[2]);
    min_p = std::stof(args[3]);
    max_p = std::stof(args[4]);
    no_click_cost = std::stof(args[5]);
    click_cost = std::stof(args[6]);
    p_strategy = std::stoi(args[7]);
    tot_iter = std::stoi(args[8]);
    mod_iter = std::stoi(args[9]);
    rnd_seed = std::stoi(args[10]);

    run(ml_args, tot_iter, mod_iter, rnd_seed, num_contexts, num_actions, min_p, max_p, no_click_cost, click_cost, p_strategy);
}
