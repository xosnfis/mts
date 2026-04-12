FROM ghcr.io/open-webui/pipelines:main

# Install all Python dependencies at build time
RUN pip install --no-cache-dir \
    numpy \
    httpx \
    python-pptx \
    lxml \
    pypdf \
    python-docx \
    pandas \
    openpyxl \
    unstructured \
    fpdf2 \
    tabulate

# Copy utility modules
COPY file_processor.py /app/utils/file_processor.py
COPY file_generator.py /app/utils/file_generator.py

# Copy pipeline files
COPY router_pipeline.py /app/pipelines/router_pipeline.py
COPY advanced_memory_pipeline.py /app/pipelines/advanced_memory_pipeline.py
COPY image_gen_pipeline.py /app/pipelines/image_gen_pipeline.py
COPY web_search_pipeline.py /app/pipelines/web_search_pipeline.py
COPY deep_research_pipeline.py /app/pipelines/deep_research_pipeline.py
COPY pptx_pipeline.py /app/pipelines/pptx_pipeline.py
