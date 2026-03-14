# Stage 1 (~2 min):
nohup /media/skr/storage/conda_envs/action100m/bin/python /media/skr/storage/3DGS/RhodusAI/Action100M/tests/test_stage1_short.py > /media/skr/storage/3DGS/RhodusAI/Action100M/tests/logs/test_stage1_short.log 2>&1 &



# Stage 2 — Leaf captioning:                                                               
nohup python tests/test_stage2_leaf.py > tests/logs/stage2_leaf.log 2>&1 &
                                                                                                                                                
# Stage 2 — Non-leaf captioning:                            
nohup python tests/test_stage2_nonleaf.py > tests/logs/stage2_nonleaf.log 2>&1 &




# Stage 3 (run after both Stage 2s done):
OPENAI_API_KEY="sk-proj-fVXNCzL7UD4UqEt8N4VlVSadzKBVETtWfgagHvBB_rf3kS97HI
4cdl8UnSKZVFMmv74Kh--wJ5T3BlbkFJpaLDpoI6qI5w0JKPHRk3Rag_Doom3IR4jhGASvTJND
5RvPwFPnpbvZmSt-Vz_LzFzxPYRcgLoA"

nohup python tests/test_stage3.py > tests/logs/stage3.log 2>&1 &