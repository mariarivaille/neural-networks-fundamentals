import torch


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            gradient_reversal_strength=1.0,
            activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
        ):
        super().__init__()

        noise_dim = channels[0]
        image_dim = channels[-1]
        generator_sizes = channels[1:]
        discriminator_sizes = channels[-2::-1]

        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
        self.gradient_reversal = self.generator_discriminator_bridge

        generator_layers = []
        current = noise_dim
        for i, size in enumerate(generator_sizes):
            generator_layers.append(torch.nn.Linear(current, size))
            if i < len(generator_sizes) - 1:
                generator_layers.append(activation())
            current = size
        generator_layers.append(torch.nn.Tanh())
        self.generator = torch.nn.Sequential(*generator_layers)

        discriminator_layers = []
        current = image_dim
        for i, size in enumerate(discriminator_sizes):
            discriminator_layers.append(torch.nn.Linear(current, size))
            if i < len(discriminator_sizes) - 1:
                discriminator_layers.append(activation())
            current = size
        self.discriminator = torch.nn.Sequential(*discriminator_layers)

        self.classifier = torch.nn.Linear(noise_dim, 1)

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        noise = batch['data']['noise']
        real = batch['data'].get('real', batch['data'].get('image'))

        generated = self.generator(noise)
        reversed_generated = self.generator_discriminator_bridge(generated)

        if real is not None:
            real_flat = real.reshape(real.shape[0], -1)
            B = noise.shape[0]
            discriminator_input = torch.cat([reversed_generated, real_flat], dim=0)
            logits = self.classifier(self.discriminator(discriminator_input)).flatten()
            fake_logits = logits[:B]
            real_logits = logits[B:]
        else:
            logits = self.classifier(self.discriminator(reversed_generated)).flatten()
            fake_logits = logits
            real_logits = None

        batch['signals'] = {
            'generated': generated,
            'discriminator_logits': logits,
            'fake_logits': fake_logits,
            'discriminator_scores': logits,
            'fake_scores': fake_logits,
        }
        batch['postprocessed'] = {
            'discriminator_score': logits,
            'fake_score': fake_logits,
            'discriminator_probability': torch.sigmoid(logits),
            'fake_probability': torch.sigmoid(fake_logits),
        }

        if real_logits is not None:
            batch['signals']['real_logits'] = real_logits
            batch['signals']['real_scores'] = real_logits
            batch['postprocessed']['real_score'] = real_logits
            batch['postprocessed']['real_probability'] = torch.sigmoid(real_logits)

        return batch
