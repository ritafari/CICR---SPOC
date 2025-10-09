import cv2
import numpy as np

class TableProcessor:
    def __init__(self):
        self.min_table_area_ratio = 0.2  # Table must cover at least 20% of page
        self.min_cells = 4               # Need at least 4 cells
        self.min_internal_lines = 2      # Need at least 2 internal lines (horizontal or vertical)
    
    def _is_table(self, gray_image):
        """Detect if content is a table with rectangular border and internal grid lines"""
        print("    🔍 Looking for table structure with border and grid...")
        
        # Create binary image
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find the main rectangular border (table outline)
        table_border = self._find_table_border(binary)
        if not table_border:
            print("    ❌ No table border found")
            return False
        
        # Check if we have internal grid lines inside the border
        has_internal_grid = self._has_internal_grid_lines(binary, table_border)
        
        # Check if we have cell structure
        has_cells = self._has_cell_structure(binary, table_border)
        
        print(f"    📊 Table check - Border: ✓, Internal grid: {has_internal_grid}, Cells: {has_cells}")
        
        return has_internal_grid and has_cells
    
    def _find_table_border(self, binary_image):
        """Find the main rectangular border that could be a table outline"""
        # Find all contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = binary_image.shape
        total_area = height * width
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - table border should be reasonably large
            if area < total_area * self.min_table_area_ratio:
                continue
            
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 sides)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's reasonably proportioned (not too skinny)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:
                    print(f"    🟦 Found table border: {w}x{h} pixels")
                    return (x, y, w, h)
        
        return None
    
    def _has_internal_grid_lines(self, binary_image, table_border):
        """Check for horizontal and vertical lines inside the table border"""
        x, y, w, h = table_border
        
        # Extract the table region
        table_region = binary_image[y:y+h, x:x+w]
        if table_region.size == 0:
            return False
        
        # Find horizontal lines inside the table
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        horizontal_lines = cv2.morphologyEx(table_region, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find vertical lines inside the table
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical_lines = cv2.morphologyEx(table_region, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count significant internal lines (avoid counting the border)
        h_lines = self._count_internal_lines(horizontal_lines, min_length=w*0.5)
        v_lines = self._count_internal_lines(vertical_lines, min_length=h*0.5)
        
        print(f"    📐 Internal lines - Horizontal: {h_lines}, Vertical: {v_lines}")
        
        # Need at least some internal lines to form a grid
        return (h_lines + v_lines) >= self.min_internal_lines
    
    def _count_internal_lines(self, line_image, min_length):
        """Count lines that are likely internal grid lines (not borders)"""
        contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        internal_lines = 0
        for contour in contours:
            # Get line dimensions
            line_x, line_y, line_w, line_h = cv2.boundingRect(contour)
            line_length = max(line_w, line_h)
            
            # Check if it's long enough to be a grid line
            if line_length >= min_length:
                internal_lines += 1
        
        return internal_lines
    
    def _has_cell_structure(self, binary_image, table_border):
        """Check if the table has clear cell structure"""
        x, y, w, h = table_border
        
        # Extract table region
        table_region = binary_image[y:y+h, x:x+w]
        if table_region.size == 0:
            return False
        
        # Find closed contours inside the table (potential cells)
        contours, hierarchy = cv2.findContours(table_region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        cell_count = 0
        table_area = w * h
        
        for i, contour in enumerate(contours):
            # Only look at inner contours (child contours in hierarchy)
            if hierarchy[0][i][3] != -1:  # Has parent (is inside another contour)
                area = cv2.contourArea(contour)
                
                # Filter by area - cells should be reasonable size
                if area < 100 or area > table_area * 0.3:
                    continue
                
                # Check if it's roughly rectangular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.2 < circularity < 0.8:  # Rectangular but not perfect
                        cell_count += 1
        
        print(f"    🟦 Found {cell_count} table cells")
        return cell_count >= self.min_cells
    
    def extract_table_structure(self, image_region, ocr_results):
        """Extract table structure only if it's definitively a table with border and grid"""
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        
        if not self._is_table(gray):
            return "Regular document text (no table structure with border and grid detected)"
        
        print("    ✅ Processing as structured table...")
        
        # Process as table
        table_lines = self._group_ocr_results_by_lines(ocr_results)
        formatted_table = self._format_table_lines(table_lines)
        
        return formatted_table
    
    def _group_ocr_results_by_lines(self, ocr_results):
        """Group OCR results into lines for table reconstruction"""
        if not ocr_results:
            return []
        
        # Sort by y-coordinate
        results_sorted = sorted(ocr_results, key=lambda x: x[0][0][1])
        
        lines = []
        current_line = []
        line_threshold = 15
        
        for bbox, text, confidence in results_sorted:
            if not current_line:
                current_line.append((bbox, text, confidence))
            else:
                # Get average y of current line
                current_y = np.mean([point[1] for bbox_curr, _, _ in current_line for point in bbox_curr])
                # Get y of current word
                current_word_y = np.mean([point[1] for point in bbox])
                
                # If y difference is small, same line
                if abs(current_word_y - current_y) < line_threshold:
                    current_line.append((bbox, text, confidence))
                else:
                    # New line
                    lines.append(current_line)
                    current_line = [(bbox, text, confidence)]
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _format_table_lines(self, lines):
        """Format lines into table structure"""
        formatted_lines = []
        
        for line in lines:
            # Sort words in line by x-coordinate (left to right)
            line_sorted = sorted(line, key=lambda x: x[0][0][0])
            
            # Extract text and join with pipe separator
            line_text = ' | '.join([text for _, text, _ in line_sorted if text.strip()])
            if line_text:
                formatted_lines.append(line_text)
        
        if not formatted_lines:
            return "Table detected but no readable content found"
        
        table_output = "[TABLE STRUCTURE]\n"
        table_output += '\n'.join(formatted_lines)
        table_output += "\n[END TABLE]"
        
        return table_output