# Annotation Tool — Source code

The web-based annotation platform (Fig. 1 of the paper) is a separate
repository. Clone it from:

  [ANON_HOST]/wound-annotation-tool.tar.gz

(URL provided in paper Conclusion; placeholder during double-blind review.)

Tech stack:
- Frontend: TypeScript + React + Konva (vector canvas)
- Backend: Python + FastAPI + SQLAlchemy
- Inference: PyTorch model server (loads U-Net + MiT-B5 weights)
