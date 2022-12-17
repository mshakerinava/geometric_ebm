sbatch batch_long.sh --manifold euclidean --dataset 8gaussians
sbatch batch_long.sh --manifold sphere --dataset 8gaussians
sbatch batch_long.sh --manifold torus --dataset 8gaussians

sbatch batch_long.sh --manifold euclidean --dataset 2spirals
sbatch batch_long.sh --manifold sphere --dataset 2spirals
sbatch batch_long.sh --manifold torus --dataset 2spirals

sbatch batch_long.sh --manifold euclidean --dataset checkerboard
sbatch batch_long.sh --manifold sphere --dataset checkerboard
sbatch batch_long.sh --manifold torus --dataset checkerboard

sbatch batch_long.sh --manifold euclidean --dataset rings
sbatch batch_long.sh --manifold sphere --dataset rings
sbatch batch_long.sh --manifold torus --dataset rings
