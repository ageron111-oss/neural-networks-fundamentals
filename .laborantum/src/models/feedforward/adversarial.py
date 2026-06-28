import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        ### YOUR CODE HERE
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
        hidden_dim = channels[1]
        output_dim = channels[2]

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(noise_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Tanh()  
        )

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(output_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, noise_dim),
            activation()
        )

        self.classifier = torch.nn.Linear(noise_dim, 1)

        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
        self.gradient_reversal = self.generator_discriminator_bridge

        ## YOUR CODE HERE

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        ## YOUR CODE HERE
        noise = batch['data']['noise']
        B = noise.shape[0]

        generated = self.generator(noise)
        real = batch['data'].get('real', batch['data'].get('image'))
        if real is not None:
            real = real.reshape(B, -1)
        
        reversed_features = self.generator_discriminator_bridge(generated)
        
        fake_features = self.discriminator(reversed_features)
        fake_logits = self.classifier(fake_features).flatten()  # (B,)
        fake_scores = torch.sigmoid(fake_logits)

        if real is not None:
            real_features = self.discriminator(real)
            real_logits = self.classifier(real_features).flatten()  # (B,)
            real_scores = torch.sigmoid(real_logits)
            
            # Конкатенируем fake и real для combined логитов
            discriminator_logits = torch.cat([fake_logits, real_logits])  # (2*B,)
            discriminator_scores = torch.cat([fake_scores, real_scores])
        else:
            real_logits = None
            real_scores = None
            discriminator_logits = fake_logits
            discriminator_scores = fake_scores
        
        batch['signals'] = {
            'generated': generated,
            'discriminator_logits': discriminator_logits,
            'fake_logits': fake_logits,
            'discriminator_scores': discriminator_scores,
            'fake_scores': fake_scores
        }
        
        if real is not None:
            batch['signals']['real_logits'] = real_logits
            batch['signals']['real_scores'] = real_scores
        
        batch['postprocessed'] = {
            'discriminator_score': discriminator_scores,
            'fake_score': fake_scores,
            'discriminator_probability': discriminator_scores,
            'fake_probability': fake_scores
        }
        
        if real is not None:
            batch['postprocessed']['real_score'] = real_scores
            batch['postprocessed']['real_probability'] = real_scores
        
        return batch