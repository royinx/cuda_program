# Docker image setting

```
docker build -t cpp_cv -f Dockerfile.opencv .
```


```bash
docker run -dit --name=cpp_cv \
		--runtime=nvidia \
		-v ~/Desktop/:/workdir \
		-w /workdir \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=unix$DISPLAY \
		cpp_cv
```
