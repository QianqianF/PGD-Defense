import torch
import torch.nn as nn

def fgsm_(model_, device, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits, _ = model_(input_)
    target = torch.LongTensor([target]).to(device)
    model_.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    # perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


# minimizes regression output
def fgsm_regression_(model_, device, x, eps, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    _, loss = model_(input_)
    loss.backward()

    out = input_ - eps * input_.grad.sign()

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def pgd_(model_, device, x, target, k, eps, eps_step, targeted=True, clip_min=None, clip_max=None, random_start=False, regression=False):
    x = x.clone().detach()

    x_min = x - eps
    x_max = x + eps

    # Randomize the starting point x.
    if random_start:
        x = x + eps * (2 * torch.rand_like(x) - 1)
        if (clip_min is not None) or (clip_max is not None):
            x.clamp_(min=clip_min, max=clip_max)

    xs = [x.clone().detach()]
    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None)
        # as we want to apply the attack as defined
        if regression:
            x = fgsm_regression_(model_, device, x, eps)
        else:
            x = fgsm_(model_, device, x, target, eps_step, targeted)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)
        xs.append(x.clone().detach())
    # if desired clip the output back to the image domain
    xs = torch.stack(xs)
    if (clip_min is not None) or (clip_max is not None):
        xs.clamp_(min=clip_min, max=clip_max)
    return xs[:, 0, :, :, :]