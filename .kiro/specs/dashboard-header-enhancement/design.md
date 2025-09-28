# Design Document

## Overview

The dashboard header enhancement will create a visually appealing, branded header section that establishes L'Oréal's identity and provides clear context for the TrendSpotter analytics dashboard. The design leverages the existing L'Oréal brand styling already implemented in the dashboard and incorporates the available logo asset.

## Architecture

The header will be implemented as a Streamlit container with custom HTML/CSS styling that integrates with the existing brand theme. The implementation will use:

- Streamlit's `st.markdown()` with `unsafe_allow_html=True` for custom HTML structure
- CSS styling that extends the existing L'Oréal brand variables and classes
- Base64 encoding for the logo image to ensure reliable display
- Responsive design principles using CSS flexbox for layout

## Components and Interfaces

### Header Container Component
- **Purpose**: Main container that holds all header elements
- **Structure**: HTML div with custom CSS classes
- **Styling**: Uses L'Oréal brand colors and existing CSS variables
- **Positioning**: Fixed at the top of the dashboard, above existing content

### Logo Component
- **Purpose**: Display the L'Oréal logo prominently
- **Implementation**: Base64-encoded image embedded in HTML
- **Sizing**: Responsive height (40-60px) with maintained aspect ratio
- **Positioning**: Left-aligned within the header container

### Title Section Component
- **Purpose**: Display dashboard title and team information
- **Structure**: Hierarchical text elements (main title + subtitle)
- **Typography**: Uses Arial font family consistent with existing theme
- **Content**: "TrendSpotter" as main title, "AI Glow-Rithms" as team subtitle

### Styling Integration
- **Brand Colors**: Utilizes existing CSS variables (--loreal-red, --loreal-black, --loreal-white)
- **Typography**: Consistent with existing font definitions
- **Spacing**: Follows established padding and margin patterns
- **Shadows**: Subtle box-shadow for visual separation

## Data Models

### Logo Asset Handling
```python
# Base64 encoding function for logo
def encode_logo_to_base64(logo_path: str) -> str
    # Returns base64 string for embedding in HTML
```

### Header Configuration
```python
# Configuration object for header customization
header_config = {
    "logo_path": "loreal-logo.jpeg",
    "main_title": "TrendSpotter",
    "subtitle": "AI Glow-Rithms",
    "height": "80px",
    "background_color": "#FFFFFF"
}
```

## Error Handling

### Logo Loading Failures
- **Scenario**: Logo file not found or corrupted
- **Handling**: Display header with text-only fallback, show warning message
- **Fallback**: Use L'Oréal text logo or company name

### CSS Styling Issues
- **Scenario**: CSS conflicts with existing Streamlit styles
- **Handling**: Use specific CSS selectors and !important declarations where necessary
- **Fallback**: Graceful degradation to basic styling

### Responsive Design Failures
- **Scenario**: Header doesn't display properly on mobile devices
- **Handling**: CSS media queries for different screen sizes
- **Fallback**: Stack elements vertically on small screens

## Testing Strategy

### Visual Testing
- **Cross-browser compatibility**: Test in Chrome, Firefox, Safari, Edge
- **Responsive design**: Test on desktop, tablet, and mobile viewports
- **Brand consistency**: Verify colors match L'Oréal brand guidelines

### Integration Testing
- **Streamlit compatibility**: Ensure header doesn't interfere with existing dashboard functionality
- **CSS integration**: Verify no conflicts with existing styles
- **Performance**: Check that base64 logo doesn't significantly impact load times

### User Experience Testing
- **Accessibility**: Verify proper alt text for logo, sufficient color contrast
- **Navigation**: Ensure header doesn't obstruct dashboard controls
- **Professional appearance**: Validate that header enhances brand perception

## Implementation Details

### CSS Structure
```css
.loreal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
    background-color: var(--loreal-white);
    border-bottom: 2px solid var(--loreal-red);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.loreal-logo {
    height: 50px;
    width: auto;
}

.loreal-title-section {
    text-align: center;
    flex-grow: 1;
}
```

### HTML Structure
```html
<div class="loreal-header">
    <img src="data:image/jpeg;base64,{base64_logo}" class="loreal-logo" alt="L'Oréal Logo">
    <div class="loreal-title-section">
        <h1>TrendSpotter</h1>
        <p>AI Glow-Rithms</p>
    </div>
    <div class="loreal-spacer"></div>
</div>
```

### Responsive Breakpoints
- **Desktop (>1024px)**: Full horizontal layout with logo left, title center
- **Tablet (768px-1024px)**: Slightly reduced padding and font sizes
- **Mobile (<768px)**: Stacked layout with centered elements