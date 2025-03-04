import bayesian

def bnn_linear_layer(params, d):
    layer_type = d.__class__.__name__ + "Reparameterization"
    layer_fn = getattr(bayesian, layer_type)  # Get BNN layer
    bnn_layer = layer_fn(
        in_features=d.in_features,
        out_features=d.out_features,
        bias=d.bias is not None,
        decay=params["decay"],
        sigma_init=params["sigma_init"],
    )

    if params["pretrain"]:
        bnn_layer.mu_weight.data.copy_(d.weight.data)
        bnn_layer.prior_mu_weight.data.copy_(d.weight.data)
        if bnn_layer.bias:
            bnn_layer.mu_bias.data.copy_(d.bias.data)
            bnn_layer.prior_mu_bias.data.copy_(d.bias.data)
    return bnn_layer


def bnn_conv_layer(params, d):
    layer_type = d.__class__.__name__ + "Reparameterization"
    layer_fn = getattr(bayesian, layer_type)  # Get BNN layer
    bnn_layer = layer_fn(
        in_channels=d.in_channels,
        out_channels=d.out_channels,
        kernel_size=d.kernel_size,
        stride=d.stride,
        padding=d.padding,
        dilation=d.dilation,
        groups=d.groups,
        bias=d.bias is not None,
        decay=params["decay"],
        sigma_init=params["sigma_init"],
    )

    if params["pretrain"]:
        bnn_layer.mu_weight.data.copy_(d.weight.data)
        bnn_layer.prior_mu_weight.data.copy_(d.weight.data)
        if bnn_layer.bias:
            bnn_layer.mu_bias.data.copy_(d.bias.data)
            bnn_layer.prior_mu_bias.data.copy_(d.bias.data)
    return bnn_layer


def bnn_ln_layer(params, d):
    layer_type = d.__class__.__name__ + "Reparameterization"
    layer_fn = getattr(bayesian, layer_type)  # Get BNN layer
    bnn_layer = layer_fn(
        normalized_shape=d.normalized_shape,
        eps=d.eps,
        bias=d.bias is not None,
        decay=params["decay"],
        sigma_init=params["sigma_init"],
    )

    if params["pretrain"]:
        bnn_layer.mu_weight.data.copy_(d.weight.data)
        bnn_layer.prior_mu_weight.data.copy_(d.weight.data)
        if bnn_layer.bias:
            bnn_layer.mu_bias.data.copy_(d.bias.data)
            bnn_layer.prior_mu_bias.data.copy_(d.bias.data)
    return bnn_layer

def bnn_in_layer(params, d):
    layer_type = d.__class__.__name__ + "Reparameterization"
    layer_fn = getattr(bayesian, layer_type)  # Get BNN layer
    bnn_layer = layer_fn(
        num_features=d.num_features,
        eps=d.eps,
        momentum=d.momentum,
        affine=True,
        track_running_stats=d.track_running_stats,
        bias=True,
        decay=params["decay"],
        sigma_init=params["sigma_init"],
    )

    if params["pretrain"]:
        bnn_layer.mu_weight.data.copy_(d.weight.data)
        bnn_layer.prior_mu_weight.data.copy_(d.weight.data)
        if bnn_layer.bias:
            bnn_layer.mu_bias.data.copy_(d.bias.data)
            bnn_layer.prior_mu_bias.data.copy_(d.bias.data)
    return bnn_layer


def convert2bnn_selective(model, config):
    for name, module in model.named_modules():
        if getattr(module, 'bayesian', False):
            convert2bnn(module, config)

def convert2bnn(m, config):
    for name, value in list(m._modules.items()):
        if m._modules[name]._modules:
            convert2bnn(m._modules[name], config)
        elif "Linear" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_linear_layer(config, m._modules[name]))
        elif "Conv" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_conv_layer(config, m._modules[name]))
        else:
            pass
    return

# in V2 we also convert LayerNorm to Baye
def convert2bnnv2(m, config):
    for name, value in list(m._modules.items()):
        if m._modules[name]._modules:
            convert2bnnv2(m._modules[name], config)
        elif "Linear" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_linear_layer(config, m._modules[name]))
        elif "Conv" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_conv_layer(config, m._modules[name]))
        elif "LayerNorm2d" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_ln_layer(config, m._modules[name]))
        elif "InstanceNorm2d" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_in_layer(config, m._modules[name]))
        else:
            pass
    return

def set_prediction_type(model, deterministic=True):
    if deterministic:
        for name, module in model.named_modules():
            if hasattr(module, 'deterministic'):
                module.deterministic = True
    else:
        for name, module in model.named_modules():
            if hasattr(module, 'deterministic'):
                module.deterministic = False


def get_kl_loss(m):
    kl_loss = None
    for layer in m.modules():
        if hasattr(layer, "kl_loss"):
            if kl_loss is None:
                kl_loss = layer.kl_loss()
            else:
                kl_loss += layer.kl_loss()
    return kl_loss
