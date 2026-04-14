# export WANDB_DISABLED=true

deepspeed --include localhost:0,1,2,3,4,5 reward_modeling_zj1.py \
    --deepspeed dpsd_zero2.json \
    --model_name_or_path /njfs/train-nlp/huipeng6/weights/Qwen2.5-7B-Instruct \
    --dataset_name preference_data_for_r1_0319.jsonl \
    --output_dir outputs/Qwen2-32B-Instruct-Reward-test001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 200 \
    --eval_strategy steps \
    --eval_steps 400 \
    --save_strategy steps \
    --save_steps 4000 \
    --max_length 2048 \
    --save_total_limit 3 \
    --report_to wandb \
    --torch_dtype bfloat16 \
    --bf16 True





(base) root@iv-ycit3j84338rx6hmd8i1:/njfs/train-nlp/huzheng/debug_train_rm# head -1 preference_data_for_r1_0319.jsonl | jq .
{
  "chosen": [
    {
      "role": "user",
      "content": "c#"
    },
    {
      "role": "assistant",
      "content": "C#（发音为\"C sharp\"）是由微软开发的一种现代、面向对象的编程语言。它被广泛用于构建各种类型的应用程序，包括网页应用、桌面应用、移动应用和游戏。C#与其它编程语言如Java和C++类似，以其简单性和易用性而闻名。C#是一种强大的语言，提供了一套丰富的库和框架，使构建稳健和可扩展的应用程序变得容易。\n\n以下是C#的一些关键特性的简要概述：\n\n1. 面向对象：C#是一种面向对象的语言，这意味着它使用对象的概念来表示现实世界的实体及其行为。\n\n2. 跨平台：C#可用于为多个平台构建应用程序，包括Windows、macOS和Linux。\n\n3. 强类型：C#是一种强类型语言，这意味着变量必须声明为特定类型，其类型在运行时不能更改。\n\n4. 事件驱动：C#使用事件驱动的编程模型，这意味着程序是围绕事件的概念构建的，如用户输入或网络活动。\n\n5. 垃圾回收：C#具有垃圾回收器，可自动管理内存分配和释放，使编写内存高效和稳健的应用程序变得更容易。\n\n6. 社区驱动：C#拥有一个庞大且活跃的开发者社区，他们通过开源项目和其他倡议为语言及其库做出贡献。\n\n总的来说，C#是一种功能强大且多用途的编程语言，被广泛用于构建各种应用程序。"
    }
  ],
  "rejected": [
    {
      "role": "user",
      "content": "c#"
    },
    {
      "role": "assistant",
      "content": "C# 是一种高级的、面向对象的编程语言，由 Microsoft 作为其 .NET 计划的一部分开发。它被创建为对 Java 的现代替代，支持多种编程范式，包括命令式、函数式和事件驱动。C# 主要用于 Windows 应用程序开发，但也可用于 Web、移动和游戏开发。该语言设计为安全、可靠和高效，为开发人员提供了丰富的库和工具，用于构建强大且可扩展的应用程序。C# 在游戏开发行业也得到了广泛应用，特别是在 Xbox 360 和 Xbox One 游戏机的游戏开发中。"
    }
  ]
}
(base) root@iv-ycit3j84338rx6hmd8i1:/njfs/train-nlp/huzheng/debug_train_rm# wc preference_data_for_r1_0319.jsonl
    79734  40808345 408692025 preference_data_for_r1_0319.jsonl
