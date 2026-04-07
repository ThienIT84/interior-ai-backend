#!/usr/bin/env python3
"""
Script to update router.py to include SAM3 endpoint
"""

router_path = "backend/app/api/v1/router.py"

# Read current content
with open(router_path, 'r') as f:
    content = f.read()

# Check if sam3_endpoint already imported
if 'sam3_endpoint' in content:
    print("✅ SAM3 endpoint already imported")
else:
    # Update import
    old_import = "from app.api.v1.endpoints import health, segmentation, inpainting, generation"
    new_import = "from app.api.v1.endpoints import health, segmentation, inpainting, generation, sam3_endpoint"
    content = content.replace(old_import, new_import)
    
    # Add router
    old_router = 'api_router.include_router(segmentation.router, prefix="/segmentation", tags=["segmentation"])'
    new_router = '''api_router.include_router(segmentation.router, prefix="/segmentation", tags=["segmentation"])
api_router.include_router(sam3_endpoint.router, prefix="/sam3", tags=["sam3"])'''
    content = content.replace(old_router, new_router)
    
    # Write back
    with open(router_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated router.py with SAM3 endpoint")
