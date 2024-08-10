import os
import torch


from .thirdparty.LoRA.gpt import get_gpt2_config, GPT2LMModel
from .misc import print_rank_0
from .thirdparty.qlora.llama import get_llama_model_and_tokenizer


def load_model_and_tokenizer(args):
    if "gpt2" in args.model_name_or_path:
        config = get_gpt2_config(args)

        model = GPT2LMModel(config)
        model.load_weight(torch.load(args.model_name_or_path))
        tokenizer = None
    elif "llama" in args.model_name_or_path:
        model, tokenizer = get_llama_model_and_tokenizer(args)
    else:
        raise NotImplementedError

    return model, tokenizer


def save_model(args, model, tokenizer, lr_scheduler, optimizer):
    print_rank_0("saving model ...", args.global_rank)
    model_path = os.path.join(args.output_dir, "model.pt")
    state_dict = model.state_dict()
    for n, m in model.named_modules():
        if args.scope in n and m.__class__.__name__ == "BlockSparseLinear":
            if "gpt" in args.model_name_or_path:
                state_dict[n + ".weight"] = m.w0.squeeze().t()
            else:
                state_dict[n + ".weight"] = m.w0.squeeze()
            state_dict[n + ".bias"] = m.b

    torch.save({"model_state_dict": state_dict}, model_path)
    torch.save(tokenizer, os.path.join(args.output_dir, "tokenizer.pt"))
    torch.save(
        lr_scheduler.state_dict(), os.path.join(args.output_dir, "lr_scheduler.pt")
    )
    torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
    torch.cuda.empty_cache()
    print("done")
