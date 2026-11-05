LARGE ARTIFACTS — Copy / Restore Instructions
===============================================

Mục đích
--------
Tệp này liệt kê các thư mục/tệp lớn trong workspace mà hiện không nên đẩy lên Git (được ignore) và cần được copy thủ công lên máy hoặc một storage ngoài (S3 / HF / artifact storage) để chạy hệ thống. File này cũng chỉ dẫn cách restore thủ công các dữ liệu này vào tree làm việc local khi cần.

Ghi chú chung
-------------
- `.gitignore` trong repository đã thiết lập để không theo dõi các thư mục/tệp lớn (models, embed_data, out_lora_adapter, v.v.).
- Thay đổi `.gitignore` chỉ ảnh hưởng tới các tệp mới được thêm. Nếu các tệp lớn đã nằm trong lịch sử Git, bạn cần xóa/migrate chúng khỏi lịch sử trước khi push (xem phần "Xử lý file lớn đã commit" ở cuối).
- Nên lưu các artefact lớn (mô hình, embeddings) trên storage ngoài như Git LFS, S3, hoặc Hugging Face Hub, không commit trực tiếp vào Git.

Tóm tắt các đường dẫn lớn (tìm từ hệ thống)
-------------------------------------------
(Dưới đây là kết quả kiểm tra hiện tại trên workspace của bạn.)

- `models/`
  - exists: YES
  - tracked by git: NO (không có file tracked hiện tại dưới path này)
  - size on disk: ~21G
  - notes: chứa các model `.safetensors` rất lớn (ví dụ `model-00002-of-00002.safetensors` ~2.4GB). Không commit trực tiếp.

- `embed_data/`
  - exists: YES
  - tracked by git: NO
  - size on disk: ~192M
  - notes: chứa JSON embeddings/metadata dùng cho retriever; copy khi cần chạy local retrieval.

- `embed/`
  - exists: YES
  - tracked by git: SOME FILES tracked (ví dụ `embed/em_lesson.py`, `embed/test_embed_qa.py`)
  - size on disk: ~8.0K
  - notes: mã hỗ trợ; không phải là dữ liệu lớn.

- `out_lora_adapter/`
  - exists: YES
  - tracked by git: NO
  - size on disk: ~31M
  - notes: chứa adapter weights (safetensors). Nếu cần inference với LoRA, copy thủ công.

- `PL_dialogue/`
  - exists: YES
  - tracked by git: (no files currently tracked under this path)
  - size on disk: ~67M
  - notes: chứa `PLGTDB_emb.json` (~66M) — lớn, nên lưu ngoài Git.

- `data/`
  - exists: YES
  - tracked by git: SOME FILES tracked (ví dụ `data/dialogue/*.json`, `data/qa600.json`)
  - size on disk: ~2.5M
  - notes: tập dữ liệu nhỏ (một phần đã tracked). Những file lớn trong `data/` nếu có sẽ cần lưu ngoài git.

- `PL_dialogue/PLGTDB_emb.json`
  - exists: YES
  - tracked by git: (no)
  - size on disk: ~66M

Hướng dẫn copy thủ công (từ nơi lưu trữ ngoài vào máy dev)
---------------------------------------------------------
Giả sử bạn có các artefact ở một nơi an toàn (S3, NAS, hoặc một máy khác). Dưới đây là ví dụ các lệnh copy cơ bản (chạy trong thư mục repo root):

- Copy toàn bộ `models/`:

```bash
# ví dụ copy từ /mnt/storage/models
rsync -avh --progress /mnt/storage/models/ ./models/
```

- Copy `embed_data/`:

```bash
rsync -avh --progress /mnt/storage/embed_data/ ./embed_data/
```

- Copy `out_lora_adapter/`:

```bash
rsync -avh --progress /mnt/storage/out_lora_adapter/ ./out_lora_adapter/
```

- Copy PL dialogue embed file:

```bash
cp /mnt/storage/PLGTDB_emb.json PL_dialogue/PLGTDB_emb.json
```

Lưu ý an toàn sau khi copy
--------------------------
- Không chạy `git add models` hay `git add embed_data` nếu `.gitignore` đang chặn các đường dẫn này. Nếu bạn muốn tạm thời commit một tệp để debug, cân nhắc dùng một branch tạm và xóa tệp sau đó, nhưng tốt nhất là tránh commit file lớn.
- Nếu bạn cần chia sẻ models cho team, lưu trên storage trung tâm (S3, HF, Artifactory) và chia sẻ link thay vì commit vào Git.

Xử lý các file lớn đã từng được commit (nếu có)
-----------------------------------------------
Nếu trước đây bạn đã lỡ commit các file lớn và chúng nằm trong lịch sử Git (làm cho push thất bại), bạn cần rewrite lịch sử để loại bỏ chúng. Có hai cách phổ biến:

1) Dùng Git LFS + migrate (nếu bạn muốn giữ các file trong LFS):

```bash
# cài và bật LFS
git lfs install
# track các pattern lớn
git lfs track "*.safetensors" "embed_data/**" "PL_dialogue/**"
# commit .gitattributes
git add .gitattributes && git commit -m "Track large files with LFS"
# migrate history (very slow on big repos) — rewrite history
git lfs migrate import --include="*.safetensors,embed_data/**,PL_dialogue/**,out_lora_adapter/**" --everything --yes
# force push (remote sẽ cần reset)
git push origin --all --force
```

2) Dùng BFG hoặc git-filter-repo để xóa hoàn toàn file/folder khỏi lịch sử (destructive):

```bash
# tạo mirror an toàn
git clone --mirror . ../repo-mirror.git
cd ../repo-mirror.git
# dùng BFG (xóa folder 'models' và embed_data)
java -jar /path/to/bfg.jar --delete-folders models --delete-folders embed_data --delete-files '*.safetensors' .
# dọn dẹp và push
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```

Cảnh báo: cả hai cách đều rewrite lịch sử. Trước khi làm: backup repo, thông báo cho team, và chuẩn bị cho mọi người re-clone hoặc hard-reset.

Gợi ý lưu trữ an toàn cho models / embeddings
--------------------------------------------
- Hugging Face Hub (model + tokenizer + config) — hữu ích nếu mô hình được dùng chung
- S3 / MinIO / NAS cho artifacts nội bộ
- GitHub Releases (upload binary artifacts bên ngoài commit)
- Git LFS (nhưng cần lưu ý quota và chi phí)

Kết luận ngắn gọn
-----------------
- Các thư mục/tệp lớn cần copy thủ công: `models/` (~21GB), `embed_data/` (~192MB), `out_lora_adapter/` (~31MB), `PL_dialogue/PLGTDB_emb.json` (~66MB).
- Không cố gắng commit trực tiếp các file này — dùng Git LFS hoặc external storage.
- Nếu các file này đã từng được commit trong lịch sử, bạn cần rewrite lịch sử (BFG/git-filter-repo) hoặc migrate sang LFS trước khi push.

Nếu bạn muốn, mình có thể:
- Tạo script nhỏ `scripts/restore_artifacts.sh` chứa các lệnh rsync/cp mẫu.
- Thực hiện migration sang Git LFS (mình sẽ chạy `git lfs migrate import ...` nếu bạn đồng ý rewrite history và force-push).
- Hoặc tạo branch `clean-for-remote` chỉ chứa code (mình có thể push branch đó giúp bạn).

Cho mình biết bạn muốn mình làm bước tiếp theo nào (tạo script restore, migrate sang LFS, hay tạo branch clean và push).
