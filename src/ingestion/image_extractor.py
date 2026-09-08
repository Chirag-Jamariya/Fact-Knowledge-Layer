"""Image and visual chart extractor module using PyMuPDF (fitz)."""

import logging
from typing import List, Optional
import pymupdf as fitz
from src.models.corpus import ExtractedImage

logger = logging.getLogger(__name__)


def extract_images_from_fitz_page(
    page: fitz.Page,
    page_number: int,
    min_width: int = 120,
    min_height: int = 120,
) -> List[ExtractedImage]:
    """
    Extract embedded images and figures from a PyMuPDF page.
    Filters out small icons, glyphs, and line separators.
    """
    extracted_images: List[ExtractedImage] = []

    try:
        image_list = page.get_images(full=True)
        if not image_list:
            return extracted_images

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = page.parent.extract_image(xref)
            if not base_image:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            image_format = base_image.get("ext", "png")

            # Filter out tiny decorative icons / logos
            if width < min_width or height < min_height:
                continue

            # Attempt to locate surrounding caption text
            caption = None
            try:
                # Get text around the image location if rect is available
                rects = page.get_image_rects(xref)
                if rects:
                    img_rect = rects[0]
                    # Expand rect slightly downward to capture typical figure captions
                    caption_rect = fitz.Rect(
                        img_rect.x0,
                        img_rect.y1,
                        img_rect.x1,
                        min(img_rect.y1 + 40, page.rect.y1),
                    )
                    surrounding_text = page.get_text("text", clip=caption_rect).strip()
                    if surrounding_text:
                        caption = " ".join(surrounding_text.split())
            except Exception:
                pass

            extracted_images.append(
                ExtractedImage(
                    page_number=page_number,
                    image_index=img_idx,
                    caption=caption or f"Figure/Image {img_idx+1} on Page {page_number}",
                    width=width,
                    height=height,
                    image_format=image_format,
                )
            )
    except Exception as e:
        logger.warning(f"Error extracting images from page {page_number}: {e}")

    return extracted_images
