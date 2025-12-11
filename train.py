import argparse
import os
from data_loader import load_trajectories_npz, build_synthetic_demo
from agent import A2CAgent
from tqdm import tqdm
import numpy as np
import random
import torch

def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def train_main(args):
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    # Load dataset: replace with your own file
    if args.data is None:
        print("No dataset provided. Using synthetic demo dataset - replace with preprocessed AmsterdamUMCdb trajectories.")
        trajs = build_synthetic_demo(num_traj=500, n_features=args.input_dim)
    else:
        trajs = load_trajectories_npz(args.data)

    # split (paper: 70/20/10)
    random.shuffle(trajs)
    n = len(trajs)
    train_trajs = trajs[:int(0.7*n)]
    val_trajs = trajs[int(0.7*n):int(0.9*n)]
    test_trajs = trajs[int(0.9*n):]

    agent = A2CAgent(input_dim=args.input_dim, n_actions=5, device=device,
                     lr=args.lr, value_coef=args.value_coef,
                     entropy_coef=args.entropy_coef, gamma=args.gamma)

    best_val_loss = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        random.shuffle(train_trajs)
        losses = []
        for traj in tqdm(train_trajs, desc=f"Epoch {epoch}", leave=False):
            r = agent.update_from_trajectory(traj)
            losses.append(r['loss'])
        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch}/{args.epochs} train_loss={mean_loss:.5f}")

        # validation (simple pass to measure value/policy losses)
        if epoch % args.val_every == 0:
            val_losses = []
            for traj in val_trajs:
                with torch.no_grad():
                    states = torch.tensor(traj['states'], dtype=torch.float32, device=device)
                    logits, values = agent.model(states)
                    # simple value error vs returns
                    values_np = values.detach().cpu().numpy()
                    next_v = 0.0
                    returns, _ = agent.compute_returns_and_advantages(traj['rewards'], traj['dones'], values_np, next_v)
                    val_losses.append(float(((values_np - returns)**2).mean()))
            val_loss = float(np.mean(val_losses))
            print(f"Validation MSE value: {val_loss:.5f}")
            ckpt_path = os.path.join(args.out_dir, f'ckpt_epoch{epoch}.pth')
            agent.save(ckpt_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                agent.save(os.path.join(args.out_dir, 'best_model.pth'))
    print("Training complete. Best val loss:", best_val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='path to .npz with trajectories (optional).')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='where to save models')
    parser.add_argument('--input_dim', type=int, default=379, help='number of input features (paper: 379)')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    train_main(args)
