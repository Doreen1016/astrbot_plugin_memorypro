import json
import os
import httpx
import re
from datetime import datetime
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register 
from astrbot.core.platform import AstrMessageEvent
from astrbot.api.provider import LLMResponse
from astrbot.api import logger

@register("MemoryPro", "DD", "核心记忆系统：全域联想与高健壮版", "1.1.1")
class MemoryPlugin(Star):
    
    def __init__(self, context: Context, config: dict = None, **kwargs):
        super().__init__(context)
        self.conf = config if config else {}
        base_path = os.path.abspath(os.path.join(os.getcwd(), "data", "memory"))
        self.memory_dir = base_path + os.sep
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)
        
        self.temp_history = {} 
        self.counters = {} 

    async def get_persona_id(self, event: AstrMessageEvent) -> str:
        # 优先读取配置中的人格名称
        fixed = self.conf.get("fixed_name", "").strip()
        return fixed if fixed else "人格"

    @filter.on_llm_request()
    async def record_user_and_inject(self, event: AstrMessageEvent, req): 
        if str(event.get_sender_id()) == str(event.get_self_id()): return
        sid = event.session_id
        if sid not in self.temp_history: self.temp_history[sid] = []
        
        user_msg = event.get_message_str()
        user_name = event.get_sender_name() or "你"
        self.temp_history[sid].append({"role": user_name, "content": user_msg})
        
        persona_name = await self.get_persona_id(event)
        filename = f"{self.memory_dir}{persona_name}_{event.get_sender_id()}.json"
        
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data:
                    # 1. 基础记忆：提取最近 3 条维持当下语境
                    relevant = data[-3:]
                    
                    # 2. 全域联想：提取关键词并扫描整本 JSON 档案
                    potential_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}', user_msg)
                    stop_words = {"什么", "可以", "怎么", "我们", "你们", "一个", "觉得", "那个", "这个", "东西"}
                    clean_keywords = [w for w in potential_keywords if w not in stop_words]

                    if clean_keywords:
                        search_pool = data[:-3] # 在旧档案中扫描
                        matches = []
                        for entry in search_pool:
                            score = sum(1 for kw in clean_keywords if kw in entry['summary'])
                            if score > 0: matches.append((score, entry))
                        
                        # 按匹配权重排序，取最相关的 5 条旧记忆
                        matches.sort(key=lambda x: x[0], reverse=True)
                        for _, entry in matches[:5]:
                            if entry not in relevant: relevant.append(entry)

                    # 3. 按时间重排并注入系统提示词
                    relevant.sort(key=lambda x: x['timestamp'])
                    memories_text = [f"[{e['timestamp']}] {e['summary']}" for e in relevant]
                    inject_text = f"\n\n【核心记忆档案馆】\n{persona_name} 的历史记录要点：\n" + "\n".join(memories_text)
                    if hasattr(req, 'system_message'):
                        req.system_message = (req.system_message or "") + inject_text
            except Exception as e:
                logger.error(f"[MemoryPro] 检索异常: {e}")

    @filter.on_llm_response()
    async def record_ai_and_sum(self, event: AstrMessageEvent, response: LLMResponse):
        if str(event.get_sender_id()) == str(event.get_self_id()): return
        sid = event.session_id
        if sid not in self.temp_history: self.temp_history[sid] = []
        self.temp_history[sid].append({"role": "我", "content": response.completion_text})
        
        self.counters[sid] = self.counters.get(sid, 0) + 1
        if self.counters[sid] >= self.conf.get("threshold", 100):
            await self._generate_summary(sid, event.get_sender_id(), await self.get_persona_id(event), event.get_sender_name())
            self.counters[sid] = 0

    @filter.command("强制总结")
    async def force_summarize(self, event: AstrMessageEvent):
        persona_name = await self.get_persona_id(event)
        user_name = event.get_sender_name() or "伙伴"
        yield event.plain_result(f"🚀 正在提炼核心记忆...")
        await self._generate_summary(event.session_id, event.get_sender_id(), persona_name, user_name)

    async def _generate_summary(self, sid, user_id, persona_name, user_name):
        try:
            history = self.temp_history.get(sid, [])
            if not history: return
            history_text = "\n".join([f"[{m['role']}]: {m['content']}" for m in history])
            
            prompt = (
                f"你现在是 {persona_name}。请以第一人称（我）的视角，为你和 {user_name} 的互动写一条日记。\n"
                f"【强制约束】：\n"
                f"1. 严禁使用'用户'、'对方'、'TA'等词汇。\n"
                f"2. 请直接从对话记录中观察你对她的专属称呼，并以此称呼她。\n"
                f"3. 保持你的性格特征。禁止废话，120字以内。\n\n"
                f"对话记录：\n{history_text}"
            )
            
            api_url, api_key = self.conf.get("api_url", ""), self.conf.get("api_key", "")
            summary = ""
            
            if api_url and api_key:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{api_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": self.conf.get("model_name", "gpt-4o-mini"), "messages": [{"role": "user", "content": prompt}]},
                        timeout=30.0
                    )
                    # ✨ 修复点：增加 choices 存在性检查，防止 API 报错导致崩溃
                    res_data = resp.json()
                    if resp.status_code == 200 and 'choices' in res_data:
                        summary = res_data['choices'][0]['message']['content']
                    else:
                        logger.error(f"[MemoryPro] API 响应异常: {res_data}")
            else:
                # ✨ 修复点：适配 v4.22.3 的接口调用方式，替换掉 get_main_llm
                try:
                    prov = self.context.get_using_provider()
                    if prov:
                        res = await prov.text_chat(prompt)
                        summary = res.completion_text
                    else:
                        logger.error("[MemoryPro] 无法获取主 LLM 驱动。")
                except Exception as inner_e:
                    logger.error(f"[MemoryPro] 调用主模型失败: {inner_e}")

            if summary:
                clean_summary = re.sub(r"^(好的|这是总结).*?[:：\s\n]*", "", summary, flags=re.S).strip()
                filename = f"{self.memory_dir}{persona_name}_{user_id}.json"
                self._save_to_json(filename, clean_summary)
                logger.info(f"[MemoryPro] 记忆已静默入库。")
                self.temp_history[sid] = []
        except Exception as e: 
            logger.error(f"[MemoryPro] 总结系统故障: {e}")

    def _save_to_json(self, filename, summary):
        entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "summary": summary}
        data = []
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except: data = []
        data.append(entry)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)