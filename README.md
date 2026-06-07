# wexon-i2v-worker

WEXON T2 image-to-video GPU worker (RunPod Serverless).

Оживляет картинку (FLUX-сцену) в короткий клип через **LTX-Video** (лицензия
OpenRAIL-M — коммерческое использование разрешено).

## Deploy (RunPod Serverless, deploy-from-GitHub)
1. New Endpoint → Import from GitHub → этот репозиторий, ветка `main`.
2. GPU: 24 GB (RTX 4090 / L4 / A5000 / L40). Active Workers = 0 (scale-to-zero).
3. Env vars:
   - `CLOUDINARY_URL=cloudinary://<key>:<secret>@<cloud>`
   - (опц.) `LTX_MODEL=Lightricks/LTX-Video` — модель можно сменить без пересборки.
4. (Прод) Attach Network Volume → mount `/runpod-volume`, чтобы модель LTX
   кэшировалась между холодными стартами (иначе качается заново).

## API
`POST /run` → `{"input": {"image_url": "...", "prompt": "...", "width":480,
"height":832, "num_frames":97, "steps":40, "fps":24, "seed":0}}`
→ `{"video_url": "<cloudinary>", "frames": n, "size":[w,h], "took": sec}`
