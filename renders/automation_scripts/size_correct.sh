cd ~/renders/automation_scripts

echo "Starting trianglesplatting..." && bash render_trianglesplatting.sh && \
echo "Starting radegs_official..." && bash render_radegs_official.sh && \
echo "Starting 3dcs..." && bash render_3dcs.sh && \
echo "Starting betasplatting..." && bash render_betasplatting.sh && \
echo "All renders completed successfully."