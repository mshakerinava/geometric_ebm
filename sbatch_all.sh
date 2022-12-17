sbatch batch_long.sh --epochs 5 --manifold euclidean --dataset 8gaussians
sbatch batch_long.sh --epochs 5 --manifold sphere --dataset 8gaussians
sbatch batch_long.sh --epochs 5 --manifold torus --dataset 8gaussians

sbatch batch_long.sh --epochs 5 --manifold euclidean --dataset 2spirals
sbatch batch_long.sh --epochs 5 --manifold sphere --dataset 2spirals
sbatch batch_long.sh --epochs 5 --manifold torus --dataset 2spirals

sbatch batch_long.sh --epochs 5 --manifold euclidean --dataset checkerboard
sbatch batch_long.sh --epochs 5 --manifold sphere --dataset checkerboard
sbatch batch_long.sh --epochs 5 --manifold torus --dataset checkerboard

sbatch batch_long.sh --epochs 5 --manifold euclidean --dataset rings
sbatch batch_long.sh --epochs 5 --manifold sphere --dataset rings
sbatch batch_long.sh --epochs 5 --manifold torus --dataset rings
