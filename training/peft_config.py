from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    IA3Config,
    PromptEncoderConfig,
    AdaLoraConfig,
    BottleneckConfig,
)


def get_peft(model, peft):
    """
    load peft configuration
    """
    task_type = "CAUSAL_LM"
    peft_mapping = {
        "lora": (
            LoraConfig,
            {
                "task_type": task_type,
                "inference_mode": False,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["c_proj", "c_attn", "q_attn"],
            },
        ),
        "ia3": (
            IA3Config,
            {
                "task_type": task_type,
                "inference_mode": False,
                "target_modules": ["c_attn", "mlp.c_proj"],
                "feedforward_modules": ["mlp.c_proj"],
            },
        ),
        "ptuning": (
            PromptEncoderConfig,
            {
                "task_type": task_type,
                "inference_mode": False,
                "num_virtual_tokens": 30,
            },
        ),
        "adapterh": (
            BottleneckConfig,
            {
                "task_type": task_type,
                "bottleneck_size": 256,
                "non_linearity": "tanh",
                "adapter_dropout": 0.0,
                "use_parallel_adapter": False,
                "use_adapterp": False,
                "target_modules": ["c_fc", "mlp.c_proj"],
                "scaling": 1.0,
            },
        ),
        "adapterp": (
            BottleneckConfig,
            {
                "task_type": task_type,
                "bottleneck_size": 256,
                "non_linearity": "tanh",
                "adapter_dropout": 0.0,
                "use_parallel_adapter": False,
                "use_adapterp": True,
                "target_modules": ["mlp.c_proj"],
                "scaling": 1.0,
            },
        ),
        "parallel": (
            BottleneckConfig,
            {
                "task_type": task_type,
                "bottleneck_size": 256,
                "non_linearity": "tanh",
                "adapter_dropout": 0.0,
                "use_parallel_adapter": True,
                "use_adapterp": False,
                "target_modules": ["c_attn", "q_attn"],
                "scaling": 1.0,
            },
        ),
        # "adalora": (
        #     AdaLoraConfig,
        #     {
        #         "task_type": task_type,
        #         "inference_mode": False,
        #         "init_r": 12,
        #         "target_r": 8,
        #         "beta1": 0.85,
        #         "beta2": 0.85,
        #         "tinit": 200,
        #         "tfinal": 1000,
        #         "deltaT": 10,
        #         "lora_alpha": 16,
        #         "lora_dropout": 0.05,
        #         "target_modules": ["c_proj", "c_attn", "q_attn"],
        #     },
        # ),
        # "prompt": (
        #     PromptTuningConfig,
        #     {"task_type": task_type, "inference_mode": False, "num_virtual_tokens": 30},
        # ),
    }
    config_class, config_kwargs = peft_mapping.get(peft, (None, None))
    if config_class:
        print(config_class(**config_kwargs))
        # for peft tuning
        return get_peft_model(model, config_class(**config_kwargs))
    else:
        # return model for full finetune
        return model
