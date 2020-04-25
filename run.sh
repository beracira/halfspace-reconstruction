docker build -t halfspace .
docker run -it --rm --name runner --mount type=bind,source=$(pwd),target=/app halfspace
