# Adarsh Kandoth - Professional Portfolio

A modern, minimalist portfolio website showcasing professional experience, projects, and skills in business analytics and data science. Features interactive project case studies with professional visualizations and downloadable PDF reports.

[![Portfolio](https://img.shields.io/badge/Portfolio-Live-gold)](https://your-portfolio-url.com)
[![License](https://img.shields.io/badge/License-Personal-blue)](LICENSE)

---

## 🌟 Features

- **Responsive Design**: Fully responsive across all devices (desktop, tablet, mobile)
- **Modern UI**: Clean, elegant design with neutral color scheme (black, white, gold accents)
- **Interactive Projects**: Detailed case studies with modal popups
- **Professional Visualizations**: 12 data visualization charts across 6 projects
- **PDF Reports**: Downloadable project reports (41.5MB of detailed documentation)
- **Professional Headshot**: High-quality photo in hero section
- **Contact Form**: Integrated email contact functionality
- **Smooth Animations**: Scroll-based animations and transitions
- **SEO Optimized**: Proper meta tags and semantic HTML
- **Fast Loading**: Optimized assets (~70MB total, efficiently organized)

---

## 🛠️ Tech Stack

- **HTML5**: Semantic markup for accessibility and SEO
- **CSS3**: Modern styling with CSS Grid, Flexbox, and CSS Variables
- **JavaScript (Vanilla)**: No frameworks or libraries required - lightweight and fast
- **Font Awesome**: Icon library for professional iconography
- **Python**: Used for generating professional data visualizations (Matplotlib, Seaborn)

---

## 📄 Sections

1. **Hero Section**: Professional introduction with headshot and key statistics
2. **About**: Professional summary and career highlights
3. **Experience**: Interactive timeline of professional roles (8+ years)
4. **Projects**: Six featured projects with detailed case studies, visualizations, and PDF reports
5. **Skills**: Categorized skill sets and education
6. **Contact**: Contact information and email form

---

## 🎯 Projects Showcased

### 1. Healthcare Data Analysis for Predictive Diabetes Insights
- **Accuracy**: 99.89% with Tuned Random Forest
- **Course**: DS-542 Python in Data Science
- **Visualizations**: Model comparison charts, EDA histograms
- **PDF Report**: Available (3.2MB)

### 2. Employee Attrition Prediction Using Machine Learning
- **Accuracy**: 92.95% with Random Forest
- **Dataset**: IBM HR Analytics (1,470 records, 35 features)
- **Visualizations**: Model performance comparison, attrition by department
- **PDF Report**: Available (5.3MB)

### 3. Zomato Delivery Route Optimization & ETA Prediction
- **Accuracy**: 92.95% with Random Forest
- **Dataset**: 45,584 delivery orders
- **Visualizations**: Correlation heatmaps, traffic impact analysis
- **PDF Report**: Available (12MB)

### 4. Portfolio Analysis – ELF Cosmetics
- **Focus**: DCF Valuation and Financial Modeling
- **Course**: GB-530 Corporate Finance
- **Visualizations**: Revenue trends, DCF valuation scenarios
- **PDF Report**: Not available (proprietary analysis)

### 5. Airbnb Sales Analysis | Power BI Dashboard
- **Platform**: Power BI with DAX
- **Course**: DS-621 Business Analytics and Power BI
- **Visualizations**: Revenue analysis, occupancy trends (10+ visuals)
- **PDF Report**: Available (21MB)

### 6. Critical Analysis of Research Bias and Methodology
- **Focus**: Anthropomorphism in Behavioral Science
- **Course**: DS-520 Probability and Statistics
- **Visualizations**: Research methodology framework
- **PDF Report**: Available (91KB)

---

## 🚀 Quick Start

### Local Development

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/yourusername/adarsh-portfolio.git
   cd adarsh-portfolio
   ```

2. **Open in browser**
   - Simply double-click `index.html`, or
   - Use a local server (recommended):
   ```bash
   python3 -m http.server 8000
   # Open http://localhost:8000
   ```

3. **No build process required** - Pure HTML/CSS/JavaScript

---

## 🌐 Deployment Options

### Option 1: GitHub Pages (Recommended)

**Quick Deploy - Automated Script:**
```bash
# Simply run the deployment script
./deploy-to-github.sh

# The script will guide you through:
# 1. Initialize git repository
# 2. Create initial commit
# 3. Push to GitHub
# 4. Enable GitHub Pages
```

**Manual Deployment:**
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial portfolio deployment"

# Create repository on GitHub, then:
git remote add origin https://github.com/yourusername/portfolio.git
git branch -M main
git push -u origin main

# Enable GitHub Pages in repository settings
# Your site will be live at: https://yourusername.github.io/portfolio
```

**📖 See [GITHUB-PAGES-DEPLOY.md](GITHUB-PAGES-DEPLOY.md) for detailed instructions, troubleshooting, and custom domain setup.**

### Option 2: Netlify (Easiest)

1. Go to [netlify.com](https://netlify.com)
2. Sign up for free account
3. Drag and drop the entire project folder
4. Site deployed instantly at `https://your-name.netlify.app`

### Option 3: Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Follow prompts - site live in seconds
```

### Option 4: Traditional Web Hosting

1. Upload all files via FTP/SFTP
2. Ensure `index.html` is in the root directory
3. Access via your domain


---

## 📁 File Structure

```
adarsh-portfolio/
├── index.html                              # Main portfolio page
├── README.md                               # This file - complete documentation
├── Adarsh Kandoth Updated V12012026.pdf   # Professional resume
│
└── assets/
    ├── css/
    │   └── styles.css                      # All styling (responsive, animations)
    │
    ├── js/
    │   └── script.js                       # All functionality (modals, navigation)
    │
    ├── images/
    │   ├── adarsh-photo.png               # Professional headshot (976KB)
    │   └── projects/                       # Project visualizations (12 images)
    │       ├── healthcare/
    │       │   ├── model-comparison.png
    │       │   └── eda-distributions.png
    │       ├── attrition/
    │       │   ├── model-performance.png
    │       │   └── attrition-by-department.png
    │       ├── zomato/
    │       │   ├── model-accuracy.png
    │       │   ├── correlation-heatmap.png
    │       │   └── traffic-impact.png
    │       ├── portfolio/
    │       │   ├── revenue-trend.png
    │       │   └── dcf-valuation.png
    │       ├── airbnb/
    │       │   ├── revenue-by-type.png
    │       │   └── occupancy-trends.png
    │       └── research/
    │           └── research-framework.png
    │
    └── pdfs/
        └── projects/                       # Detailed project reports
            ├── healthcare-diabetes-prediction.pdf      (3.2MB)
            ├── employee-attrition.pdf                  (5.3MB)
            ├── zomato-delivery-optimization.pdf        (12MB)
            ├── airbnb-sales-analysis.pdf               (21MB)
            └── research-paper-review.pdf               (91KB)
```


---

## 🎨 Customization

### Updating Personal Information

**1. Hero Section** ([index.html](index.html))
```html
<h1 class="hero-title">Your Name</h1>
<p class="hero-description">Your Title</p>
```

**2. Contact Information** ([index.html](index.html))
- Email, phone, LinkedIn, location in contact section
- Update `mailto:` link in contact form

**3. Resume**
- Replace `Adarsh Kandoth Updated V12012026.pdf` with your resume
- Update filename in download button

**4. Professional Photo**
- Replace `assets/images/adarsh-photo.png` with your headshot
- Recommended size: 800x800px, optimized for web

### Updating Projects

**1. Project Data** ([assets/js/script.js](assets/js/script.js))

Modify the `projectData` object:
```javascript
projectData = {
    yourproject: {
        title: 'Your Project Title',
        subtitle: 'Project Category',
        course: 'Course Name',
        overview: 'Project overview...',
        challenge: 'Problem statement...',
        solution: 'Your solution...',
        technologies: ['Tech1', 'Tech2'],
        keyFeatures: ['Feature 1', 'Feature 2'],
        results: ['Result 1', 'Result 2'],
        impact: 'Business impact...',
        visualizations: ['Chart 1', 'Chart 2'],
        images: [
            { src: 'assets/images/projects/yourproject/chart1.png', alt: 'Description' }
        ],
        pdf: 'assets/pdfs/projects/yourproject-report.pdf'
    }
}
```

**2. Project Card** ([index.html](index.html))

Add project cards to the projects section:
```html
<div class="project-card">
    <div class="project-icon"><i class="fas fa-icon"></i></div>
    <h3>Project Title</h3>
    <p class="project-subtitle">Category</p>
    <p>Description</p>
    <div class="project-metrics">
        <div class="metric">
            <span class="metric-value">99%</span>
            <span class="metric-label">Accuracy</span>
        </div>
    </div>
    <button onclick="openProjectModal('yourproject')">View Case Study</button>
</div>
```

**3. Add Visualizations**
- Create folder: `assets/images/projects/yourproject/`
- Add PNG files (300 DPI recommended)
- Update image paths in JavaScript

**4. Add PDF Reports**
- Add PDF: `assets/pdfs/projects/yourproject-report.pdf`
- Update path in JavaScript `projectData`

### Customizing Colors

**Edit CSS Variables** ([assets/css/styles.css](assets/css/styles.css)):

```css
:root {
    --primary-color: #1a1a1a;      /* Black */
    --secondary-color: #ffffff;     /* White */
    --accent-color: #d4af37;        /* Gold */
    --accent-light: #e6c968;        /* Light Gold */
    --text-dark: #333333;           /* Dark Gray */
    --text-light: #666666;          /* Medium Gray */
    --bg-light: #f8f9fa;            /* Light Background */
}
```

---

## 📊 Project Visualizations

All visualizations are professionally designed using:
- **Matplotlib**: Python plotting library
- **Seaborn**: Statistical data visualization
- **Design**: Consistent gold accent color scheme
- **Quality**: 300 DPI for crisp display
- **Format**: PNG with optimized file sizes

### Creating New Visualizations

Use Python to generate charts:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Create chart
fig, ax = plt.subplots(figsize=(10, 6))
# ... your plotting code ...

# Save
plt.savefig('assets/images/projects/yourproject/chart.png',
            dpi=300, bbox_inches='tight')
```

---

## 📧 Contact Form

The contact form currently uses a `mailto:` link for simplicity.

**Current behavior**: Opens user's email client with pre-filled information.

### Upgrading to Production Form

For production use, integrate a form service:

**Option 1: Formspree** (Recommended)
```html
<form action="https://formspree.io/f/YOUR_ID" method="POST">
    <!-- form fields -->
</form>
```

**Option 2: EmailJS**
- Sign up at [emailjs.com](https://emailjs.com)
- Add EmailJS SDK
- Configure email template

**Option 3: Custom Backend**
- Create API endpoint
- Update form handler in `assets/js/script.js`

---

## 🌐 Browser Support

Tested and optimized for:
- ✅ Chrome (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Edge (latest)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

---

## ⚡ Performance

- **Lightweight**: No heavy frameworks (pure HTML/CSS/JavaScript)
- **Fast Loading**: Minimal external dependencies
- **Optimized Assets**:
  - Images: Compressed PNG files
  - PDFs: Organized separately for on-demand loading
  - CSS/JS: Minification-ready
- **Responsive**: Mobile-first design approach
- **Lazy Loading**: Ready for implementation

### Performance Metrics

Test with:
- [Google PageSpeed Insights](https://pagespeed.web.dev)
- [GTmetrix](https://gtmetrix.com)

---

## 🎓 Skills Demonstrated

This portfolio showcases:

### Technical Skills
- HTML5/CSS3/JavaScript
- Responsive Web Design
- Data Visualization (Matplotlib, Seaborn)
- Python Programming
- Git Version Control

### Data Science & Analytics
- Machine Learning (Random Forest, Logistic Regression)
- Statistical Analysis
- Data Preprocessing & Feature Engineering
- Business Intelligence (Power BI, DAX)
- Financial Modeling (DCF, WACC)

### Professional Skills
- Project Documentation
- Technical Communication
- Problem Solving
- Attention to Detail

---

## 📚 Documentation

This README contains complete documentation including:

- Portfolio overview and features
- Quick start guide
- Deployment instructions (GitHub Pages, Netlify, Vercel)
- File structure and organization
- Customization guide
- Project visualization guide
- Contact form options

---

## 🔮 Future Enhancements

Potential additions:

- [ ] Blog section for articles and insights
- [ ] Dark mode toggle
- [ ] Interactive data visualizations (D3.js)
- [ ] Testimonials section
- [ ] Google Analytics integration
- [ ] Advanced contact form with backend
- [ ] Image lightbox/zoom functionality
- [ ] Video project walkthroughs
- [ ] GitHub repository links per project
- [ ] Custom domain with SSL
- [ ] Multilingual support

---

## 📄 License

Personal portfolio - All rights reserved by Adarsh Kandoth

---

## 📞 Contact

- **Email**: [adarshkandoth.us@gmail.com](mailto:adarshkandoth.us@gmail.com)
- **Phone**: [551-358-1796](tel:551-358-1796)
- **LinkedIn**: [linkedin.com/in/adarsh-kandoth-947152103](https://www.linkedin.com/in/adarsh-kandoth-947152103/)
- **Location**: New Jersey, NJ

---

## 🙏 Acknowledgments

Built with:
- ❤️ Attention to detail
- 🎨 Modern web standards
- 📊 Data-driven insights
- ✨ Professional polish

---

**Last Updated**: February 22, 2026
**Version**: 2.0
**Status**: Production-Ready

---

*Ready to impress recruiters and hiring managers with a professional, data-driven portfolio!* 🚀
