import defaultSettings from '@/settings'

const title = defaultSettings.title || '文本综合处理系统'

export default function getPageTitle(pageTitle) {
  if (pageTitle) {
    return `${pageTitle} - ${title}`
  }
  return `${title}`
}
