// Llama.cpp Turbo Desktop — Landing Page Logic

document.addEventListener('DOMContentLoaded', () => {
  // Smooth scroll for nav links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        e.preventDefault();
        targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Fetch GitHub Stars & Release info dynamically if online
  const repoName = 'Protik1810/llamacpp-turbo';
  const starBadge = document.getElementById('gh-star-count');

  if (starBadge) {
    fetch(`https://api.github.com/repos/${repoName}`)
      .then(res => res.json())
      .then(data => {
        if (data.stargazers_count !== undefined) {
          starBadge.textContent = `${data.stargazers_count} ★`;
        }
      })
      .catch(() => {
        starBadge.textContent = '★ Star';
      });
  }

  // Interactive Theme Preview Switcher (Dark / Light)
  const themeBtns = document.querySelectorAll('.theme-toggle-btn');
  const heroMockupImg = document.getElementById('hero-mockup-img');
  const galleryMainImg = document.getElementById('gallery-main-img');
  const galleryAboutImg = document.getElementById('gallery-about-img');

  const themeImages = {
    dark: {
      main: 'assets/screenshot-dark-main.png',
      about: 'assets/screenshot-dark-about.png'
    },
    light: {
      main: 'assets/screenshot-main.png',
      about: 'assets/screenshot-about.png'
    }
  };

  themeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const selectedTheme = btn.getAttribute('data-theme');
      if (!selectedTheme) return;

      themeBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll(`[data-theme="${selectedTheme}"]`).forEach(b => b.classList.add('active'));

      // Smooth cross-fade image swap
      if (heroMockupImg) {
        heroMockupImg.classList.add('fade-out');
        setTimeout(() => {
          heroMockupImg.src = themeImages[selectedTheme].main;
          heroMockupImg.classList.remove('fade-out');
        }, 180);
      }

      if (galleryMainImg) {
        galleryMainImg.classList.add('fade-out');
        setTimeout(() => {
          galleryMainImg.src = themeImages[selectedTheme].main;
          galleryMainImg.classList.remove('fade-out');
        }, 180);
      }

      if (galleryAboutImg) {
        galleryAboutImg.classList.add('fade-out');
        setTimeout(() => {
          galleryAboutImg.src = themeImages[selectedTheme].about;
          galleryAboutImg.classList.remove('fade-out');
        }, 180);
      }
    });
  });

  // Copy code snippet helper
  window.copySnippet = function(elementId, btn) {
    const text = document.getElementById(elementId)?.innerText;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const originalText = btn.innerHTML;
      btn.innerHTML = '✓ Copied!';
      btn.style.color = '#34d399';
      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.style.color = '';
      }, 2000);
    });
  };
});
