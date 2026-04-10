let extractedData = null;

// Load saved server URL
chrome.storage?.local?.get(['serverUrl'], (result) => {
  if (result && result.serverUrl) {
    document.getElementById('serverUrl').value = result.serverUrl;
  }
});

function saveServerUrl() {
  const url = document.getElementById('serverUrl').value.trim();
  chrome.storage?.local?.set({ serverUrl: url });
  setStatus('URL salva!', 'success');
  setTimeout(() => setStatus('Clique no botao abaixo para extrair o produto desta pagina.', 'loading'), 1500);
}

function setStatus(msg, type) {
  const el = document.getElementById('statusMsg');
  el.textContent = msg;
  el.className = 'status ' + type;
}

async function extractProduct() {
  const btn = document.getElementById('btnExtract');
  btn.disabled = true;
  btn.textContent = 'Extraindo...';
  setStatus('Analisando a pagina...', 'loading');
  document.getElementById('extractResult').style.display = 'none';

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractFromPage,
    });

    const data = results[0]?.result;

    if (!data || !data.title) {
      setStatus('Nenhum produto encontrado nesta pagina. Tente em uma pagina de produto.', 'error');
      return;
    }

    extractedData = data;
    extractedData.source_url = tab.url;

    // Show preview
    document.getElementById('prevTitle').textContent = data.title;
    document.getElementById('prevMeta').textContent =
      `${data.images.length} imagens encontradas` +
      (data.price ? ` | ${data.price}` : '');

    const descEl = document.getElementById('prevDesc');
    if (data.description) {
      const tmp = document.createElement('div');
      tmp.innerHTML = data.description;
      descEl.textContent = tmp.textContent.substring(0, 200) + (tmp.textContent.length > 200 ? '...' : '');
    } else {
      descEl.textContent = 'Sem descricao';
    }

    const imgGrid = document.getElementById('prevImages');
    imgGrid.innerHTML = '';
    const maxShow = 5;
    data.images.slice(0, maxShow).forEach(url => {
      const img = document.createElement('img');
      img.src = url;
      img.onerror = () => img.style.display = 'none';
      imgGrid.appendChild(img);
    });
    if (data.images.length > maxShow) {
      const more = document.createElement('div');
      more.className = 'more';
      more.textContent = `+${data.images.length - maxShow}`;
      imgGrid.appendChild(more);
    }

    document.getElementById('extractResult').style.display = 'block';
    setStatus(`Produto extraido: "${data.title}"`, 'success');

  } catch (err) {
    setStatus('Erro ao extrair: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Extrair Produto desta Pagina';
  }
}

// This function runs inside the web page context
function extractFromPage() {
  const result = {
    title: '',
    description: '',
    images: [],
    price: '',
  };

  // 1. Try JSON-LD structured data
  const jsonLdScripts = document.querySelectorAll('script[type="application/ld+json"]');
  for (const script of jsonLdScripts) {
    try {
      let data = JSON.parse(script.textContent);
      // Handle @graph arrays
      if (data['@graph']) {
        data = data['@graph'].find(item =>
          item['@type'] === 'Product' ||
          (Array.isArray(item['@type']) && item['@type'].includes('Product'))
        ) || data;
      }
      if (data['@type'] === 'Product' || (Array.isArray(data['@type']) && data['@type'].includes('Product'))) {
        result.title = data.name || '';
        result.description = data.description || '';
        if (data.image) {
          const imgs = Array.isArray(data.image) ? data.image : [data.image];
          imgs.forEach(img => {
            const url = typeof img === 'string' ? img : img.url;
            if (url && !result.images.includes(url)) result.images.push(url);
          });
        }
        if (data.offers) {
          const offer = Array.isArray(data.offers) ? data.offers[0] : data.offers;
          if (offer && offer.price) {
            const currency = offer.priceCurrency || 'BRL';
            result.price = `${currency} ${offer.price}`;
          }
        }
        break;
      }
    } catch (e) { /* ignore parse errors */ }
  }

  // 2. Fallback: Open Graph meta tags
  if (!result.title) {
    const ogTitle = document.querySelector('meta[property="og:title"]');
    if (ogTitle) result.title = ogTitle.content;
  }
  if (!result.description) {
    const ogDesc = document.querySelector('meta[property="og:description"]');
    if (ogDesc) result.description = ogDesc.content;
  }
  if (result.images.length === 0) {
    const ogImage = document.querySelector('meta[property="og:image"]');
    if (ogImage && ogImage.content) result.images.push(ogImage.content);
  }

  // 3. Fallback: page title
  if (!result.title) {
    const h1 = document.querySelector('h1');
    if (h1) result.title = h1.textContent.trim();
  }
  if (!result.title) {
    result.title = document.title;
  }

  // 4. Try to get product description from common selectors
  if (!result.description) {
    const descSelectors = [
      '.product-description',
      '.product__description',
      '#product-description',
      '.product-single__description',
      '[data-product-description]',
      '.woocommerce-product-details__short-description',
      '.product_description',
      '.description',
      '#tab-description',
      '.product-info-description',
    ];
    for (const sel of descSelectors) {
      const el = document.querySelector(sel);
      if (el && el.innerHTML.trim().length > 20) {
        result.description = el.innerHTML.trim();
        break;
      }
    }
  }

  // 5. Collect all product images from common selectors
  const imgSelectors = [
    '.product-gallery img',
    '.product__media img',
    '.product-single__media img',
    '.product-images img',
    '.product-photos img',
    '.woocommerce-product-gallery img',
    '.product-image img',
    '[data-product-media] img',
    '.gallery-image img',
    '.product__image img',
    '.swiper-slide img',
    '.slick-slide img',
    '.product-photo img',
  ];

  for (const sel of imgSelectors) {
    document.querySelectorAll(sel).forEach(img => {
      // Prefer data-src or data-zoom-image for high-res
      const src = img.dataset.zoomImage || img.dataset.large || img.dataset.src || img.dataset.original || img.src;
      if (src && !src.includes('placeholder') && !src.startsWith('data:') && !result.images.includes(src)) {
        result.images.push(src);
      }
    });
  }

  // 6. If still no images, grab large images from the page
  if (result.images.length === 0) {
    document.querySelectorAll('img').forEach(img => {
      const src = img.dataset.src || img.src;
      if (!src || src.startsWith('data:') || src.includes('logo') || src.includes('icon') || src.includes('avatar')) return;
      // Only include images that are reasonably large
      if ((img.naturalWidth > 150 || img.width > 150) && !result.images.includes(src)) {
        result.images.push(src);
      }
    });
  }

  // 7. For Shopify stores, try to get more data
  if (window.ShopifyAnalytics || window.Shopify) {
    try {
      const productMeta = document.querySelector('[data-product-json], #ProductJson, script.product-json');
      if (productMeta) {
        const pData = JSON.parse(productMeta.textContent);
        if (pData.title && !result.title) result.title = pData.title;
        if (pData.description && !result.description) result.description = pData.description;
        if (pData.images) {
          pData.images.forEach(img => {
            const url = typeof img === 'string' ? img : img.src;
            if (url && !result.images.includes(url)) result.images.push(url);
          });
        }
      }
    } catch (e) { /* ignore */ }
  }

  return result;
}

async function sendToApp() {
  if (!extractedData) return;

  const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  if (!serverUrl) {
    setStatus('Configure a URL do Image Tools acima.', 'error');
    return;
  }

  const btn = document.getElementById('btnSend');
  btn.disabled = true;
  btn.textContent = 'Enviando...';
  setStatus('Enviando produto para o Image Tools...', 'loading');

  try {
    // Encode data as URL parameters and open in a new tab
    const params = new URLSearchParams();
    params.set('ext_title', extractedData.title);
    params.set('ext_description', extractedData.description || '');
    params.set('ext_images', extractedData.images.join('\n'));
    params.set('ext_source', extractedData.source_url || '');

    const url = serverUrl + '/?source=shopify&' + params.toString();
    chrome.tabs.create({ url: url });

    setStatus('Produto enviado! Abrindo Image Tools...', 'success');

  } catch (err) {
    setStatus('Erro ao enviar: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Enviar para Image Tools';
  }
}
