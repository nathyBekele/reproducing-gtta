/**
 * Visitor Alert Tracker for the GTTA Syntactic Alignment demo (adapted from nathybekele.github.io).
 * IDs come from window.SITE_CONFIG (served by the Space's /config.js), so nothing is hard-coded here.
 * Dispatches an intelligent end-of-session notification with Microsoft Clarity recording,
 * academic network detection, link click tracking, scroll depth, session duration,
 * referring source, geolocation, device info, and exit reason via FormSubmit.co.
 */
(function () {
  "use strict";

  if (window.__saVisitorTrackerLoaded) return;
  window.__saVisitorTrackerLoaded = true;

  // FormSubmit endpoint (using your encrypted token to keep your personal email private)
  const SITE = window.SITE_CONFIG || {};
  const RECIPIENT_ENDPOINT = SITE.formsubmitUrl || "";
  const CLARITY_PROJECT_ID = SITE.clarityId || "";
  const IDLE_TIMEOUT_MS = 150000; // 2.5 minutes of inactivity triggers idle notification

  let sessionEmailSent = false;
  let cachedGeo = null;
  let clarityMeta = null;
  let maxScrollPercent = 0;
  const scrollMilestones = {};
  const interactedPopups = new Set();
  const clickedActions = [];
  const clickedUrls = new Set();

  // Session duration tracking
  let sessionStartTime = Date.now();
  let lastActiveTime = Date.now();
  let totalActiveMs = 0;
  let lastTickTime = Date.now();

  // Restore existing session state if user navigated between internal pages
  try {
    const savedStart = sessionStorage.getItem("sa_session_start_time");
    if (savedStart && !sessionStorage.getItem("sa_visitor_alert_sent")) {
      const parsedStart = parseInt(savedStart, 10);
      if (!isNaN(parsedStart) && Date.now() - parsedStart < 3600000) {
        sessionStartTime = parsedStart;
      }
    } else {
      sessionStorage.setItem("sa_session_start_time", String(sessionStartTime));
    }

    const savedActions = sessionStorage.getItem("sa_session_actions");
    if (savedActions) {
      const parsed = JSON.parse(savedActions);
      if (Array.isArray(parsed)) {
        parsed.forEach((a) => clickedActions.push(a));
      }
    }
  } catch (e) {}

  // Remove internal navigation flag upon arrival on this page
  try {
    sessionStorage.removeItem("sa_internal_nav");
  } catch (e) {}

  function handleExclusionParams() {
    try {
      const search = window.location.search;
      if (!search) return;
      const params = new URLSearchParams(search);
      if (params.has("exclude_me") || params.has("optout") || params.has("ignore_me") || params.has("admin_device")) {
        localStorage.setItem("nb_exclude_device", "true");
        document.cookie = "nb_exclude_device=true; max-age=315360000; path=/; SameSite=Lax";
        try {
          const url = new URL(window.location.href);
          url.searchParams.delete("exclude_me");
          url.searchParams.delete("optout");
          url.searchParams.delete("ignore_me");
          url.searchParams.delete("admin_device");
          window.history.replaceState({}, document.title, url.pathname + (url.search ? url.search : "") + url.hash);
        } catch (e) {}
        alert("✅ This device is now permanently excluded from visitor notification emails.");
      } else if (params.has("include_me") || params.has("optin") || params.has("track_me")) {
        localStorage.removeItem("nb_exclude_device");
        document.cookie = "nb_exclude_device=; max-age=0; path=/; SameSite=Lax";
        try {
          const url = new URL(window.location.href);
          url.searchParams.delete("include_me");
          url.searchParams.delete("optin");
          url.searchParams.delete("track_me");
          window.history.replaceState({}, document.title, url.pathname + (url.search ? url.search : "") + url.hash);
        } catch (e) {}
        alert("ℹ️ Tracking re-enabled: This device will now trigger visitor notifications.");
      }
    } catch (e) {}
  }

  handleExclusionParams();

  function isDeviceExcluded() {
    try {
      if (localStorage.getItem("nb_exclude_device") === "true") return true;
      if (localStorage.getItem("nathy_admin_pat")) return true;
      if (localStorage.getItem("nathy_admin_session")) return true;
      if (document.cookie.split(";").some((item) => item.trim() === "nb_exclude_device=true")) return true;
    } catch (e) {}
    return false;
  }

  window.excludeMyDevice = function () {
    try {
      localStorage.setItem("nb_exclude_device", "true");
      document.cookie = "nb_exclude_device=true; max-age=315360000; path=/; SameSite=Lax";
      alert("✅ This device is now permanently excluded from visitor notifications.");
    } catch (e) {}
  };

  window.includeMyDevice = function () {
    try {
      localStorage.removeItem("nb_exclude_device");
      document.cookie = "nb_exclude_device=; max-age=0; path=/; SameSite=Lax";
      alert("ℹ️ Visitor notifications re-enabled for this device.");
    } catch (e) {}
  };

  window.isDeviceExcludedFromTracking = isDeviceExcluded;

  function formatEATTime(date = new Date()) {
    try {
      const formatted = date.toLocaleString("en-US", {
        timeZone: "Africa/Addis_Ababa",
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
      });
      return `${formatted} EAT (UTC+3)`;
    } catch (e) {
      return `${date.toLocaleString()} (Local)`;
    }
  }

  function formatDuration(ms) {
    const totalSec = Math.max(1, Math.round(ms / 1000));
    if (totalSec < 60) {
      return `${totalSec}s`;
    }
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins}m ${secs}s`;
  }

  function shouldTrack() {
    const isTest = window.location.search.includes("test_alert=1");

    // Filter automated crawlers and headless environments
    if (navigator.webdriver && !isTest) return false;

    // Filter local development unless testing
    const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    if (isLocal && !isTest) return false;

    // Filter excluded owner/admin devices
    if (isDeviceExcluded() && !isTest) return false;

    // Filter repeated visits in the same session unless test/force
    if (sessionStorage.getItem("sa_visitor_alert_sent") && !isTest && !window.location.search.includes("force_alert=1")) {
      return false;
    }

    return true;
  }

  function getDeviceInfo() {
    const ua = navigator.userAgent || "";
    let os = "Unknown OS";
    if (/iPad|iPhone|iPod/.test(ua)) os = "iOS";
    else if (/Macintosh|Mac OS X/.test(ua)) os = "macOS";
    else if (/Windows NT/.test(ua)) os = "Windows";
    else if (/Android/.test(ua)) os = "Android";
    else if (/Linux/.test(ua)) os = "Linux";
    else if (/CrOS/.test(ua)) os = "ChromeOS";

    let browser = "Unknown Browser";
    if (/Edg\//.test(ua)) browser = "Microsoft Edge";
    else if (/Chrome\//.test(ua) && !/Edg\//.test(ua)) browser = "Chrome";
    else if (/Safari\//.test(ua) && !/Chrome\//.test(ua)) browser = "Safari";
    else if (/Firefox\//.test(ua)) browser = "Firefox";
    else if (/Opera|OPR\//.test(ua)) browser = "Opera";

    const isTouch = navigator.maxTouchPoints > 0;
    const screenWidth = window.screen.width;
    let deviceType = "Desktop";
    if (/Mobile|Android|iP(hone|od)/i.test(ua) || (isTouch && screenWidth < 768)) {
      deviceType = "Mobile";
    } else if (/iPad|Tablet/i.test(ua) || (isTouch && screenWidth >= 768 && screenWidth <= 1024)) {
      deviceType = "Tablet";
    }

    return { os, browser, deviceType };
  }

  function getScreenInfo() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const sw = window.screen.width;
    const sh = window.screen.height;
    const dpr = window.devicePixelRatio ? Math.round(window.devicePixelRatio * 100) / 100 : 1;

    let viewportCategory = "Mobile (<768px)";
    if (w >= 1380) viewportCategory = "Large Desktop (>=1380px)";
    else if (w >= 992) viewportCategory = "Laptop / Desktop (992px - 1379px)";
    else if (w >= 768) viewportCategory = "Tablet / Medium (768px - 991px)";

    return {
      resolution: `${sw} x ${sh} (DPR: ${dpr})`,
      viewport: `${w} x ${h}`,
      viewportCategory,
      isExpandedOnOpen: w >= 1380,
    };
  }

  async function fetchGeolocation() {
    if (cachedGeo) return cachedGeo;
    try {
      const res = await fetch("https://ipwho.is/");
      if (res.ok) {
        const data = await res.json();
        if (data && data.success !== false) {
          cachedGeo = {
            city: data.city || "Unknown City",
            country: data.country || "Unknown Country",
            region: data.region || "",
            isp: data.connection && data.connection.isp ? data.connection.isp : "N/A",
            org: data.connection && data.connection.org ? data.connection.org : "",
            domain: data.connection && data.connection.domain ? data.connection.domain : "",
            asn: data.connection && data.connection.asn ? String(data.connection.asn) : "",
            timezone: data.timezone && data.timezone.id ? data.timezone.id : "N/A",
          };
          syncGeoToClarity(cachedGeo);
          return cachedGeo;
        }
      }
    } catch (e) {}

    try {
      const res2 = await fetch("https://ipapi.co/json/");
      if (res2.ok) {
        const data2 = await res2.json();
        cachedGeo = {
          city: data2.city || "Unknown City",
          country: data2.country_name || "Unknown Country",
          region: data2.region || "",
          isp: data2.org || "N/A",
          org: data2.org || "",
          domain: "",
          asn: data2.asn || "",
          timezone: data2.timezone || "N/A",
        };
        syncGeoToClarity(cachedGeo);
        return cachedGeo;
      }
    } catch (e) {}

    cachedGeo = {
      city: "Unknown City",
      country: "Unknown Country",
      region: "",
      isp: "N/A",
      org: "",
      domain: "",
      asn: "",
      timezone: "N/A",
    };
    return cachedGeo;
  }

  function detectAcademicNetwork(geo) {
    if (!geo) return { isAcademic: false, badge: "No", institution: "N/A" };
    const textToScan = [geo.org || "", geo.isp || "", geo.domain || "", geo.asn || ""].join(" ").toLowerCase();
    const domain = (geo.domain || "").toLowerCase();

    // Check academic top-level or second-level domains (.edu, .ac.uk, .edu.et, etc.)
    const isEduDomain = /\.(edu|ac\.[a-z]{2,}|edu\.[a-z]{2,})$/.test(domain) || /\.(edu|ac\.[a-z]{2,}|edu\.[a-z]{2,})\./.test(domain);

    // Common academic and research network keywords across universities worldwide
    const academicKeywords = [
      "university",
      "universität",
      "universite",
      "universiteit",
      "universidad",
      "universita",
      "college",
      "institute of technology",
      "polytechnic",
      "polytechnique",
      "academy of science",
      "national lab",
      "research institute",
      "research center",
      "research council",
      "faculty of",
      "school of medicine",
      "campus network",
      "higher education",
      "academic",
      "observatory",
      "laboratory",
      "max planck",
      "inria",
      "cnrs",
      "fraunhofer",
      "cern",
      "eth zurich",
      "epfl",
      "kaist",
      "tsinghua",
      "peking univ",
      "oxford",
      "cambridge",
      "stanford",
      "harvard",
      "mit ",
      "berkeley",
      "princeton",
      "columbia",
      "carnegie mellon",
      "cmu",
      "eduroam",
      "geant",
      "switch",
      "dfn",
      "renater",
      "jisc",
      "canarie",
      "internet2",
      "cenic",
      "sunet",
      "heanet",
      "aarnet",
      "rediris",
      "garr",
      "belnet",
      "surfnet",
    ];

    const matchedKeyword = academicKeywords.find((kw) => textToScan.includes(kw));

    if (isEduDomain || matchedKeyword) {
      const institutionName = geo.org || geo.isp || geo.domain || "Academic Network";
      return {
        isAcademic: true,
        badge: "🎓 YES - Academic / University Network",
        institution: institutionName,
      };
    }

    return {
      isAcademic: false,
      badge: "No (Commercial / Consumer ISP)",
      institution: geo.isp || geo.org || "N/A",
    };
  }

  function analyzeReferrer() {
    const ref = document.referrer || "";
    const urlParams = new URLSearchParams(window.location.search);

    // Extract custom campaign / referral parameters
    const trackingTags = [];
    ["ref", "p", "prof", "c", "source", "from", "utm_source", "utm_medium", "utm_campaign"].forEach((param) => {
      const val = urlParams.get(param);
      if (val) trackingTags.push(`${param}=${val}`);
    });

    const trackingSummary = trackingTags.length > 0 ? trackingTags.join(", ") : "None";

    if (!ref) {
      return {
        sourceCategory:
          trackingTags.length > 0
            ? "📧 Email Link / Targeted Campaign (Direct or Email Client)"
            : "Direct Visit / Typed URL / Desktop Email App (Apple Mail, Outlook, Thunderbird, etc.)",
        referrerUrl: "(None / Direct)",
        trackingTags: trackingSummary,
      };
    }

    let refHostname = "";
    try {
      refHostname = new URL(ref).hostname.toLowerCase();
    } catch (e) {
      refHostname = ref.toLowerCase();
    }

    // Webmail clients
    if (refHostname.includes("mail.google.com")) {
      return {
        sourceCategory: "📧 Gmail (Webmail Link)",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("outlook.live.com") || refHostname.includes("outlook.office.com") || refHostname.includes("outlook.com")) {
      return {
        sourceCategory: "📧 Outlook / Office 365 (Webmail Link)",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("mail.yahoo.com")) {
      return {
        sourceCategory: "📧 Yahoo Mail (Webmail Link)",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("proton.me") || refHostname.includes("protonmail.com")) {
      return {
        sourceCategory: "📧 ProtonMail (Webmail Link)",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }

    // Search engines
    if (refHostname.includes("google.") && !refHostname.includes("scholar")) {
      return {
        sourceCategory: "🔍 Google Search",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("bing.com")) {
      return {
        sourceCategory: "🔍 Bing Search",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("duckduckgo.com")) {
      return {
        sourceCategory: "🔍 DuckDuckGo Search",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }

    // Professional & Academic networks
    if (refHostname.includes("linkedin.com") || refHostname.includes("lnkd.in")) {
      return {
        sourceCategory: "💼 LinkedIn",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("github.com")) {
      return {
        sourceCategory: "🐙 GitHub",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("twitter.com") || refHostname.includes("x.com") || refHostname.includes("t.co")) {
      return {
        sourceCategory: "🐦 X (Twitter)",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("scholar.google.")) {
      return {
        sourceCategory: "🎓 Google Scholar",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("arxiv.org")) {
      return {
        sourceCategory: "📄 arXiv",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }
    if (refHostname.includes("zenodo.org")) {
      return {
        sourceCategory: "📄 Zenodo",
        referrerUrl: ref,
        trackingTags: trackingSummary,
      };
    }

    return {
      sourceCategory: `🌐 External Site (${refHostname})`,
      referrerUrl: ref,
      trackingTags: trackingSummary,
    };
  }

  // Sync geolocation and referrer details into Microsoft Clarity tags
  function syncGeoToClarity(geo) {
    try {
      if (typeof window.clarity === "function") {
        const academic = detectAcademicNetwork(geo);
        const refInfo = analyzeReferrer();
        if (academic.isAcademic) {
          window.clarity("set", "institution", academic.institution || "Academic");
          window.clarity("upgrade", "academic_network_visitor");
        }
        if (geo && geo.city && geo.city !== "Unknown City") {
          window.clarity("set", "location", `${geo.city}, ${geo.country}`);
        }
        if (refInfo && refInfo.sourceCategory) {
          window.clarity("set", "referrer_source", refInfo.sourceCategory);
        }
        if (isDeviceExcluded()) {
          window.clarity("set", "admin_device", "true");
        }
      }
    } catch (e) {}
  }

  // Listen to Clarity metadata callback
  function initClarityListener() {
    try {
      if (typeof window.clarity === "function") {
        window.clarity(
          "metadata",
          function (meta) {
            if (meta && (meta.sessionId || meta.userId)) {
              clarityMeta = meta;
            }
          },
          false
        );
      }
    } catch (e) {}
  }

  // Extract Clarity session ID and direct playback URL
  function getClaritySessionInfo() {
    let userId = null;
    let sessionId = null;
    let playbackUrl = null;

    if (clarityMeta && clarityMeta.sessionId) {
      sessionId = clarityMeta.sessionId;
      userId = clarityMeta.userId || null;
      if (userId && sessionId) {
        playbackUrl = `https://clarity.microsoft.com/player/${clarityMeta.projectId || CLARITY_PROJECT_ID}/${userId}/${sessionId}`;
      }
    }

    if (!playbackUrl) {
      try {
        const cookies = document.cookie ? document.cookie.split(";") : [];
        let clck = null;
        let clsk = null;
        for (let i = 0; i < cookies.length; i++) {
          const c = cookies[i].trim();
          if (c.indexOf("_clck=") === 0) clck = decodeURIComponent(c.substring(6));
          if (c.indexOf("_clsk=") === 0) clsk = decodeURIComponent(c.substring(6));
        }
        if (clck && clsk) {
          const uId = clck.split(/[\^|]/)[0];
          const sId = clsk.split(/[\^|]/)[0];
          if (uId && sId) {
            userId = uId;
            sessionId = sId;
            playbackUrl = `https://clarity.microsoft.com/player/${CLARITY_PROJECT_ID}/${uId}/${sId}`;
          }
        }
      } catch (e) {}
    }

    const dashboardUrl = CLARITY_PROJECT_ID ? `https://clarity.microsoft.com/projects/view/${CLARITY_PROJECT_ID}/recordings` : "(Clarity not configured)";
    return {
      userId,
      sessionId,
      playbackUrl: playbackUrl || dashboardUrl,
      dashboardUrl,
      hasDirectPlayback: !!playbackUrl,
    };
  }

  // Classify and label clicked links (GTTA demo pages)
  function identifyLink(target, anchor) {
    const href = (anchor.getAttribute("href") || "").trim();
    const lowerHref = href.toLowerCase();
    const text = (anchor.innerText || anchor.textContent || "").trim().replace(/\s+/g, " ");
    if (lowerHref.endsWith(".pdf")) return { name: "📄 Opened the report (PDF)", category: "paper", url: href };
    if (lowerHref.includes("arxiv.org")) return { name: "📄 GTTA paper on arXiv", category: "paper", url: href };
    if (lowerHref.includes("github.com/r2llab")) return { name: "💻 GTTA code (r2llab/GTTA)", category: "code", url: href };
    if (lowerHref.includes("github.com")) return { name: "💻 Reproduction code (GitHub)", category: "code", url: href };
    if (lowerHref.includes("readme")) return { name: "📝 README", category: "code", url: href };
    if (lowerHref === "dg" || lowerHref.endsWith("/dg")) return { name: "🔎 Navigation: DG Inspector", category: "nav", url: href };
    if (lowerHref === "try" || lowerHref.endsWith("/try") || text.toLowerCase().includes("sa inspector")) return { name: "🧪 Navigation: SA Inspector (live)", category: "nav", url: href };
    if (lowerHref === "./" || lowerHref === "/" || text.toLowerCase() === "results") return { name: "🧭 Navigation: Results", category: "nav", url: href };
    if (lowerHref.startsWith("mailto:")) return { name: "📧 Email link", category: "contact", url: href };
    const cleanText = text.slice(0, 40) || href.slice(0, 40);
    return { name: `🔗 Link: "${cleanText}" (${href})`, category: "other", url: href };
  }

  function recordUserAction(name, category, url) {
    const elapsedMs = Date.now() - sessionStartTime;
    const elapsedStr = formatDuration(elapsedMs);
    const actionEntry = `[+${elapsedStr}] ${name}`;

    clickedActions.push(actionEntry);
    if (url) clickedUrls.add(url);

    // Save in sessionStorage so actions persist across internal page navigation
    try {
      sessionStorage.setItem("sa_session_actions", JSON.stringify(clickedActions));
    } catch (e) {}

    // Send custom event and upgrade to Microsoft Clarity
    try {
      if (typeof window.clarity === "function") {
        const eventTag = name
          .replace(/[^\w\s-]/g, "")
          .trim()
          .replace(/\s+/g, "_")
          .slice(0, 32);
        window.clarity("event", `click_${eventTag}`);
        window.clarity("set", "last_clicked", name);
        window.clarity("upgrade", "user_clicked_link");
      }
    } catch (e) {}
  }

  // Called by the pages: e.g. "▶️ Ran SA (task #12, their code, lr 0.1, 1.5B)"
  window.trackDemoAction = function (name) {
    try {
      if (!window.__saTrackerActive) return;
      markUserActive();
      recordUserAction(name, "demo", null);
    } catch (e) {}
  };

  function handleDocumentClick(e) {
    markUserActive();

    // Check for email copy button
    const copyBtn = e.target.closest('.email-copy-btn, [onclick*="copyEmail"]');
    if (copyBtn) {
      recordUserAction("📋 Copied Email to Clipboard", "contact", "mailto:natnaelbekele142@gmail.com");
      return;
    }

    // Check for anchor clicks
    const anchor = e.target.closest("a");
    if (!anchor) return;

    const info = identifyLink(e.target, anchor);
    recordUserAction(info.name, info.category, info.url);

    // Check if this is an internal link within nathybekele.github.io
    try {
      const targetUrl = new URL(anchor.href, window.location.href);
      if (
        targetUrl.origin === window.location.origin &&
        anchor.target !== "_blank" &&
        !anchor.getAttribute("download") &&
        targetUrl.pathname !== window.location.pathname
      ) {
        sessionStorage.setItem("sa_internal_nav", "true");
      }
    } catch (err) {}
  }

  function markUserActive() {
    lastActiveTime = Date.now();
  }

  function trackScroll() {
    markUserActive();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
    if (scrollHeight > 0) {
      const pct = Math.min(100, Math.round((scrollTop / scrollHeight) * 100));
      if (pct > maxScrollPercent) {
        maxScrollPercent = pct;

        // Fire Clarity milestone events at 25%, 50%, 75%, 90%
        [25, 50, 75, 90].forEach((m) => {
          if (maxScrollPercent >= m && !scrollMilestones[m]) {
            scrollMilestones[m] = true;
            try {
              if (typeof window.clarity === "function") {
                window.clarity("event", `scroll_${m}%`);
                if (m >= 50) window.clarity("upgrade", "deep_scroll");
              }
            } catch (e) {}
          }
        });
      }
    }
  }

  function markPopupInteraction(name) {
    markUserActive();
    if (interactedPopups.has(name)) return;
    interactedPopups.add(name);

    const elapsedStr = formatDuration(Date.now() - sessionStartTime);
    const label = name === "LinkedIn" ? "💼 Viewed LinkedIn Experience Card" : "🐙 Viewed GitHub Stats Card";
    clickedActions.push(`[+${elapsedStr}] ${label}`);

    try {
      if (typeof window.clarity === "function") {
        window.clarity("event", `view_popup_${name.toLowerCase()}`);
        window.clarity("upgrade", "popup_interaction");
      }
    } catch (e) {}
  }

  function attachPopupListeners() {
    const gh = document.querySelector(".github-item");
    const li = document.querySelector(".linkedin-item");

    if (gh && !gh.dataset.trackerBound) {
      gh.dataset.trackerBound = "true";
      ["mouseenter", "focusin", "touchstart", "click"].forEach((evt) => {
        gh.addEventListener(evt, () => markPopupInteraction("GitHub"), { passive: true });
      });
    }

    if (li && !li.dataset.trackerBound) {
      li.dataset.trackerBound = "true";
      ["mouseenter", "focusin", "touchstart", "click"].forEach((evt) => {
        li.addEventListener(evt, () => markPopupInteraction("LinkedIn"), { passive: true });
      });
    }
  }

  // Sends the complete session notification email
  async function dispatchFinalSessionAlert(triggerReason) {
    if (sessionEmailSent) return;

    // Filter repeated alerts in the same session unless forced
    const isTest = window.location.search.includes("test_alert=1");
    if (sessionStorage.getItem("sa_visitor_alert_sent") && !isTest && !window.location.search.includes("force_alert=1")) {
      return;
    }

    sessionEmailSent = true;
    sessionStorage.setItem("sa_visitor_alert_sent", "true");

    const totalDurationMs = Date.now() - sessionStartTime;
    const durationStr = formatDuration(totalDurationMs);
    const activeStr = formatDuration(totalActiveMs || totalDurationMs);

    const geo = cachedGeo || (await fetchGeolocation());
    const dev = getDeviceInfo();
    const scr = getScreenInfo();
    const academic = detectAcademicNetwork(geo);
    const refInfo = analyzeReferrer();
    const clarityInfo = getClaritySessionInfo();

    // Summary of clicks
    let clicksSummary = "None (Browsed page content only)";
    let subjectClicksTag = "";
    if (clickedActions.length > 0) {
      clicksSummary = `${clickedActions.length} item(s):\n` + clickedActions.map((a) => `• ${a}`).join("\n");
      const ranDemo = clickedActions.filter((a) => a.includes("Ran SA")).length;
      const hasPaper = clickedActions.some((a) => a.includes("report") || a.includes("paper"));
      if (ranDemo) {
        subjectClicksTag = ` [Ran live demo ×${ranDemo}]`;
      } else if (hasPaper) {
        subjectClicksTag = " [Opened report]";
      } else {
        subjectClicksTag = ` [${clickedActions.length} action(s)]`;
      }
    }

    // Scroll depth description
    let scrollDesc = `${maxScrollPercent}% of page`;
    if (maxScrollPercent >= 80) scrollDesc += " (Full page scrolled)";
    else if (maxScrollPercent >= 50) scrollDesc += " (Halfway scrolled)";

    // Popups viewed
    const popupsViewed = interactedPopups.size > 0 ? Array.from(interactedPopups).join(", ") : "None";

    // Informative subject line highlighting academic status, device, duration and actions
    const subjectPrefix = academic.isAcademic ? `🎓 [ACADEMIC VISITOR: ${academic.institution}]` : `Visitor: ${geo.city}, ${geo.country}`;
    const subjectLine = `[GTTA demo] ${subjectPrefix} (${dev.os} ${dev.deviceType}) - ${durationStr} session${subjectClicksTag}`;

    // Payload formatted as an HTML table for FormSubmit email
    const payload = {
      _subject: subjectLine,
      _template: "table",
      _captcha: "false",
      "🎥 Watch Clarity Recording": clarityInfo.playbackUrl,
      "Session Exit Trigger": triggerReason,
      "Session Duration": `${durationStr} (Active: ${activeStr})`,
      "Links & Actions Clicked": clicksSummary,
      "Max Scroll Depth": scrollDesc,
      "Pop-up Viewed": popupsViewed,
      "Academic / University Network": academic.isAcademic ? `🎓 YES (${academic.institution})` : `No (${geo.isp || "Commercial ISP"})`,
      "Referring Source": refInfo.sourceCategory,
      "Referrer URL": refInfo.referrerUrl,
      "URL Tracking Tags": refInfo.trackingTags,
      "Landing Page": window.location.pathname || "/",
      "Visitor Location": `${geo.city}, ${geo.region ? geo.region + ", " : ""}${geo.country}`,
      "Network / ISP": geo.isp,
      "Device & OS": `${dev.os} (${dev.browser})`,
      "Device Type": dev.deviceType,
      "Screen Resolution": scr.resolution,
      "Viewport Window": `${scr.viewport} (${scr.viewportCategory})`,
      "Visitor Timezone": geo.timezone,
      "Clarity Session ID": clarityInfo.sessionId || "Synced in dashboard",
      "Clarity Project Dashboard": clarityInfo.dashboardUrl,
      "Session Started (EAT)": formatEATTime(new Date(sessionStartTime)),
      "Recorded At (EAT)": formatEATTime(new Date()),
    };

    // Update Clarity final session metadata
    try {
      if (typeof window.clarity === "function") {
        window.clarity("set", "session_duration", durationStr);
        window.clarity("set", "scroll_depth", `${maxScrollPercent}%`);
        window.clarity("set", "total_clicks", String(clickedActions.length));
      }
    } catch (e) {}

    // Dispatch via fetch with keepalive: true to guarantee delivery on tab close / unload
    if (!RECIPIENT_ENDPOINT) return;
    try {
      fetch(RECIPIENT_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
        keepalive: true,
      }).catch(() => {});
    } catch (e) {}
  }

  function handlePageExit() {
    if (sessionEmailSent) return;

    // If the visitor simply navigated to another page on the same website, don't send exit alert yet
    if (sessionStorage.getItem("sa_internal_nav") === "true") return;

    const totalTime = Date.now() - sessionStartTime;
    const isTest = window.location.search.includes("test_alert=1") || window.location.search.includes("force_alert=1");

    // Filter instant bounces (<3s with no interactions) unless testing
    if (totalTime < 3000 && clickedActions.length === 0 && !isTest) return;

    dispatchFinalSessionAlert("Visitor Closed Page / Navigated Away");
  }

  function init() {
    if (!shouldTrack()) return;
    window.__saTrackerActive = true;

    // Prefetch geolocation early so it's synchronously available upon exit
    fetchGeolocation();

    // Listen to Clarity metadata
    initClarityListener();

    // Attach listeners for clicks and popup hovers
    document.addEventListener("click", handleDocumentClick, { capture: true });
    attachPopupListeners();

    // Re-check for popup elements in case of dynamic DOM changes
    const observer = new MutationObserver(() => {
      attachPopupListeners();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // Track scroll depth
    window.addEventListener("scroll", trackScroll, { passive: true });

    // Track active user events
    ["mousemove", "mousedown", "keydown", "touchstart"].forEach((evt) => {
      window.addEventListener(evt, markUserActive, { passive: true });
    });

    // Accumulate active duration every 1 second
    setInterval(() => {
      const now = Date.now();
      const delta = now - lastTickTime;
      lastTickTime = now;
      if (now - lastActiveTime < 5000) {
        totalActiveMs += delta;
      }
    }, 1000);

    // Idle detector: Trigger alert if visitor is inactive for 2.5 minutes
    setInterval(() => {
      if (sessionEmailSent) return;
      const now = Date.now();
      const idleTime = now - lastActiveTime;
      const totalTime = now - sessionStartTime;

      if (idleTime >= IDLE_TIMEOUT_MS && totalTime >= 10000) {
        dispatchFinalSessionAlert("Idle Inactivity (2.5+ mins without interaction)");
      }
    }, 5000);

    // Exit detectors: Trigger alert when visitor closes tab, minimizes, or leaves
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") {
        handlePageExit();
      }
    });

    window.addEventListener("pagehide", function () {
      handlePageExit();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
