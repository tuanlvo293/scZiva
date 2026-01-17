import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import NegativeBinomial
from preprocessing import reorder_gene_cov

# ----------------- Fully Connected -----------------
class ZIVAE_FC(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.mu_z = nn.Linear(hidden_dim, latent_dim)
        self.logv_z = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim * 3)
        self.input_dim = input_dim

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu, logv = self.mu_z(h), self.logv_z(h).clamp(-10, 10)
        return mu, logv

    def reparam(self, mu, logv):
        return mu + torch.randn_like(mu) * (0.5 * logv).exp()

    def decode(self, z):
        h = F.relu(self.fc3(z))
        p = self.fc4(h)
        pi = torch.sigmoid(p[:, :self.input_dim]).clamp(1e-4, 1 - 1e-4)
        mu = F.softplus(p[:, self.input_dim:2 * self.input_dim]).clamp(1e-4, 1e4)
        th = F.softplus(p[:, 2 * self.input_dim:]).clamp(1e-4, 1e4)
        return pi, mu, th

    def forward(self, x):
        mu_z, logv_z = self.encode(x)
        z = self.reparam(mu_z, logv_z)
        pi, mu, th = self.decode(z)
        return pi, mu, th, mu_z, logv_z


# ----------------- CNN -----------------
class ZIVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        k = min(64, input_dim)
        p = max(0, k - input_dim + 1)
        conv_out = (input_dim + 2 * p - k) + 1
        self.conv = nn.Conv1d(1, 1, k, 1, p)
        self.fc1 = nn.Linear(conv_out, hidden_dim)
        self.mu_z = nn.Linear(hidden_dim, latent_dim)
        self.logv_z = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim * 3)
        self.input_dim = input_dim

    def encode(self, x):
        h = F.relu(self.fc1(self.conv(x.unsqueeze(1)).squeeze(1)))
        mu, logv = self.mu_z(h), self.logv_z(h).clamp(-10, 10)
        return mu, logv

    def reparam(self, mu, logv):
        return mu + torch.randn_like(mu) * (0.5 * logv).exp()

    def decode(self, z):
        h = F.relu(self.fc3(z))
        p = self.fc4(h)
        pi = torch.sigmoid(p[:, :self.input_dim]).clamp(1e-4, 1 - 1e-4)
        mu = F.softplus(p[:, self.input_dim:2 * self.input_dim]).clamp(1e-4, 1e4)
        th = F.softplus(p[:, 2 * self.input_dim:]).clamp(1e-4, 1e4)
        return pi, mu, th

    def forward(self, x):
        mu_z, logv_z = self.encode(x)
        z = self.reparam(mu_z, logv_z)
        pi, mu, th = self.decode(z)
        return pi, mu, th, mu_z, logv_z

# ----------------- Loss functions -----------------
def zinb_nll(x, pi, mu, th):
    eps = 1e-8
    nb = NegativeBinomial(total_count=th, logits=torch.log(th + eps) - torch.log(mu + eps))
    ll_nb = nb.log_prob(x)
    ll_zero = torch.logaddexp(torch.log(pi + eps), torch.log1p(-pi + eps) + ll_nb)
    ll = torch.where(x == 0, ll_zero, torch.log1p(-pi + eps) + ll_nb)
    return -(ll).sum()

def loss_batch(pi, mu, th, x, mu_z, logv_z, w_aux):
    nll = zinb_nll(x, pi, mu, th) / x.numel()
    kl = (-0.5 * (1 + logv_z - mu_z.pow(2) - logv_z.exp()).sum()) / x.shape[0]
    recon = (1 - pi) * mu

    aux_loss = ((recon - x).pow(2)).mean()
    total_loss = nll + kl + w_aux * aux_loss

    return total_loss, {
        "total": total_loss.item(),
        "nll": nll.item(),
        "kl": kl.item(),
        "aux": aux_loss.item()
    }


# ----------------- ZIVA imputation  -----------------
def ZIVAimpute(Xmiss, seed=1, device=None, num_epochs=200,
               lr=1e-3, hidden_dim=128, latent_dim=64,
               verbose=False, use_cnn=True, tau=0.001,
               w_min=0.5, w_max=1.5, lam_reg=1e-3,
               reorder=True):

    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Reorder ---
    if reorder:
        X_sorted, order = reorder_gene_cov(Xmiss)
    else:
        X_sorted, order = Xmiss, np.arange(Xmiss.shape[1])

    X_tensor = torch.tensor(X_sorted, dtype=torch.float32, device=device)

    # --- Step 2: Model setup ---
    model = (ZIVAE if use_cnn else ZIVAE_FC)(X_tensor.size(1), hidden_dim, latent_dim).to(device)

    # Learnable auxiliary weight
    w_aux_raw = nn.Parameter(torch.tensor(0.0, device=device))
    def get_w_aux():
        return w_min + (w_max - w_min) * torch.sigmoid(w_aux_raw)
    opt = torch.optim.Adam(list(model.parameters()) + [w_aux_raw],lr=lr)

    # --- Step 3: Training ---
    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad()

        pi, mu, th, mu_z, logv_z = model(X_tensor)
        w_aux = get_w_aux()

        loss, stats = loss_batch(pi, mu, th, X_tensor, mu_z, logv_z, w_aux)
        reg = lam_reg * (w_aux - 1.0).pow(2)
        total_loss = loss + reg

        total_loss.backward()
        opt.step()

        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch:3d} | total={stats['total']:.3f} | "
                f"nll={stats['nll']:.3f} | kl={stats['kl']:.3f} | "
                f"aux={stats['aux']:.3f} | w_aux={get_w_aux().item():.3f}"
            )

    # --- Step 4: Imputation ---
    model.eval()
    with torch.no_grad():
        pi, mu, _, _, _ = model(X_tensor)
        X_recon = (1 - pi) * mu
        X_imp = X_tensor.clone()
        zero_mask = (X_tensor == 0)
        mask_est = (pi > tau) & zero_mask
        X_imp[mask_est] = X_recon[mask_est]
    X_imp_np = X_imp.cpu().numpy()

    # --- Step 5: Return to original order ---
    X_restored = np.zeros_like(X_imp_np)
    X_restored[:, order] = X_imp_np
 
    return X_restored, model